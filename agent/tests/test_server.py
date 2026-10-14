import asyncio

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer

from leanwatch.config import Config
from leanwatch.server import make_app

TOKEN = "test-token-123"
HOST = {"Host": "127.0.0.1:8765"}
ORIGIN = "http://127.0.0.1:8765"


def run(coro_fn):
    async def main():
        cfg = Config()
        cfg.port = 8765
        cfg.dry_run = True
        app = make_app(cfg, TOKEN)
        async with TestClient(TestServer(app)) as client:
            await coro_fn(client, app)
    asyncio.run(main())


def test_page_is_served_only_to_localhost_hosts():
    async def go(client, app):
        r = await client.get("/", headers=HOST)
        assert r.status == 200 and "LeanWatch" in await r.text()
        r = await client.get("/js/planner.js", headers=HOST)
        assert r.status == 200
        r = await client.get("/", headers={"Host": "evil.example:8765"})
        assert r.status == 403  # DNS rebinding blocked
    run(go)


def test_api_needs_the_token():
    async def go(client, app):
        assert (await client.get("/api/status", headers=HOST)).status == 403
        r = await client.get("/api/status", headers={**HOST, "X-LeanWatch-Token": "wrong"})
        assert r.status == 403
        r = await client.get("/api/status", headers={**HOST, "X-LeanWatch-Token": TOKEN})
        body = await r.json()
        assert r.status == 200 and body["permission"]["granted"] is False and body["dry_run"] is True
    run(go)


def test_api_rejects_other_websites():
    async def go(client, app):
        r = await client.post("/api/command", json={"text": "hi"},
                              headers={**HOST, "X-LeanWatch-Token": TOKEN, "Origin": "https://evil.example"})
        assert r.status == 403
    run(go)


def test_commands_wait_for_permission():
    async def go(client, app):
        h = {**HOST, "X-LeanWatch-Token": TOKEN}
        r = await client.post("/api/command", json={"text": "open notepad"}, headers=h)
        assert r.status == 409 and "permission" in (await r.json())["message"]
        assert (await client.post("/api/grant", json={"mouse": True, "keyboard": False}, headers=h)).status == 200
        st = await (await client.get("/api/status", headers=h)).json()
        assert st["permission"] == {"granted": True, "mouse": True, "keyboard": False}
        assert (await client.post("/api/command", json={"text": "   "}, headers=h)).status == 409
    run(go)


def test_websocket_checks_origin_and_token():
    async def go(client, app):
        with pytest.raises(aiohttp.WSServerHandshakeError):
            await client.ws_connect("/ws?token=wrong", headers={**HOST, "Origin": ORIGIN})
        with pytest.raises(aiohttp.WSServerHandshakeError):
            await client.ws_connect(f"/ws?token={TOKEN}", headers={**HOST, "Origin": "https://evil.example"})
        ws = await client.ws_connect(f"/ws?token={TOKEN}", headers={**HOST, "Origin": ORIGIN})
        first = await ws.receive_json(timeout=5)
        assert first["type"] == "status" and first["brain"] == "local" and first["api_ready"] is True
        await ws.send_json({"type": "grant", "scopes": {"mouse": True, "keyboard": True}})
        msg = await ws.receive_json(timeout=5)
        assert msg["type"] == "status" and msg["permission"]["granted"] is True
        await ws.send_json({"type": "nonsense"})
        msg = await ws.receive_json(timeout=5)
        assert msg["type"] == "error"
        await ws.close()
    run(go)
