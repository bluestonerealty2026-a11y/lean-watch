import threading
from types import SimpleNamespace as NS

import pytest

# The Claude brain is optional (requirements-claude.txt); skip these tests without it.
anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from leanwatch.brain import NOT_EXECUTED, Brain, Hooks
from leanwatch.config import Config
from leanwatch.controller import Controller, DryRunBackend
from leanwatch.permissions import Permissions
from leanwatch.screen import Geometry


def tool(name, inp, tid, toolset=None):
    return NS(type="tool_use", name=name, input=inp, id=tid, toolset_name=toolset)


def text(t):
    return NS(type="text", text=t)


class RecordingHooks(Hooks):
    def __init__(self, allow=True):
        self.said, self.actions, self.asked, self.allow = [], [], [], allow

    def narrate(self, text, kind="say"):
        self.said.append((kind, text))

    def action(self, name, inp, summary):
        self.actions.append(summary)

    def confirm(self, action, reason):
        self.asked.append(action)
        return self.allow


class FakeClient:
    """Plays back scripted responses and records each request."""

    def __init__(self, responses, fail_first_with=None):
        self.responses = list(responses)
        self.calls = []
        self.fail_first_with = fail_first_with
        self.beta = NS(messages=NS(create=self._create))
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})  # snapshot: the list keeps growing
        if self.fail_first_with is not None:
            err, self.fail_first_with = self.fail_first_with, None
            raise err
        return self.responses.pop(0)


def make_brain(client, allow=True, cfg=None):
    backend = DryRunBackend((1920, 1080))
    g = Geometry.for_screen(1920, 1080)
    perms = Permissions()
    perms.grant()
    stop = threading.Event()
    hooks = RecordingHooks(allow)
    c = Controller(backend, None, g, perms, speed=50, confirm=hooks.confirm, stop=stop, platform="linux")
    return Brain(cfg or Config(), c, hooks, stop, client=client), backend, hooks, stop


def test_batch_runs_in_order_and_echoes_toolset_name():
    brain, backend, hooks, _ = make_brain(FakeClient([]))
    results = brain.process([
        tool("narrate", {"text": "I open the File menu."}, "t1"),
        tool("left_click", {"coordinate": [100, 50]}, "t2", "computer"),
        tool("type", {"text": "notes"}, "t3", "computer"),
    ])
    assert [r["tool_use_id"] for r in results] == ["t1", "t2", "t3"]
    assert "toolset_name" not in results[0]
    assert all(r["toolset_name"] == "computer" and r["content"] == "OK" for r in results[1:])
    assert hooks.said == [("say", "I open the File menu.")]
    assert backend.log[-1] == ("write", "notes")


def test_failed_action_skips_the_rest_of_the_batch():
    brain, backend, _, _ = make_brain(FakeClient([]))
    results = brain.process([
        tool("left_click", {"coordinate": "nonsense"}, "a", "computer"),
        tool("type", {"text": "x"}, "b", "computer"),
    ])
    assert results[0]["is_error"]
    assert results[1] == {"type": "tool_result", "tool_use_id": "b", "toolset_name": "computer", "content": NOT_EXECUTED, "is_error": True}
    assert not any(e[0] == "write" for e in backend.log)


def test_declined_confirmation_blocks_following_actions():
    brain, backend, hooks, _ = make_brain(FakeClient([]), allow=False)
    results = brain.process([
        tool("request_confirmation", {"action": "Delete Old Photos", "reason": "Cannot be undone."}, "c1"),
        tool("key", {"text": "Delete"}, "c2", "computer"),
    ])
    assert results[0]["content"] == "declined"
    assert results[1]["is_error"] and "declined" in results[1]["content"]
    assert hooks.asked == ["Delete Old Photos"]
    assert backend.log == []


def test_full_run_and_request_shape():
    client = FakeClient([
        NS(stop_reason="tool_use", content=[
            NS(type="thinking", thinking="Looking for the Start button.", signature="s"),
            tool("narrate", {"text": "I click Start."}, "n1"),
            tool("left_click", {"coordinate": [20, 700]}, "k1", "computer"),
        ]),
        NS(stop_reason="end_turn", content=[text("1. Click Start.\n2. Done.")]),
    ])
    brain, backend, hooks, _ = make_brain(client)
    result = brain.run("open the start menu")
    assert result.ok and result.message.startswith("1. Click Start.")
    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["tools"][0] == {"type": "computer_toolset_20260801"}
    assert first["thinking"] == {"type": "adaptive", "display": "updates"}
    assert first["fallbacks"] == "default"
    assert set(first["betas"]) == {"thinking-display-updates-2026-08-18", "server-side-fallback-2026-07-01"}
    assert first["output_config"] == {"effort": "medium"}
    # Append-only history: user, assistant, tool results.
    second = client.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert {r["tool_use_id"] for r in second[2]["content"]} == {"n1", "k1"}
    assert ("thinking", "Looking for the Start button.") in hooks.said
    assert any(e[0] == "click" for e in backend.log)


def test_optional_betas_are_dropped_if_the_account_lacks_them():
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    bad = anthropic.BadRequestError("unknown beta", response=httpx2.Response(400, request=req), body=None)
    client = FakeClient([NS(stop_reason="end_turn", content=[text("Done.")])], fail_first_with=bad)
    brain, _, _, _ = make_brain(client)
    assert brain.run("anything").ok
    assert "betas" not in client.calls[1] and "fallbacks" not in client.calls[1]


def test_refusal_and_stop():
    brain, _, _, stop = make_brain(FakeClient([NS(stop_reason="refusal", content=[])]))
    assert not brain.run("x").ok
    brain, _, _, stop = make_brain(FakeClient([]))
    stop.set()
    r = brain.run("x")
    assert not r.ok and "Stopped" in r.message
