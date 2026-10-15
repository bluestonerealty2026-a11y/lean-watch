"""Spotting steps that are hard to undo from the model's own explanation (its Thought).

The model should call ask_user before such a step. This is the backup for when it forgets:
if the Thought says it is about to delete, send, buy, install, ... the person approves first.

The model explains in the language the person chose (--language), so the check knows the
words in several languages. For a language it doesn't know, LeanWatch fails safe and asks
before every step that clicks, drags, types or presses keys.
"""

from __future__ import annotations

import re

_PATTERNS = {
    "english": [
        r"delet\w*", r"remov\w*", r"eras\w*", r"wip(?:e|es|ed|ing)\b", r"uninstall\w*", r"install\w*",
        r"format(?:s|ting)? (?:the |this |a |your )?(?:drive|disk|partition|usb|sd card|card)",
        r"purchas\w*", r"buy\w*", r"sell\w*", r"sold\b", r"pay(?:s|ing|ment)?\b", r"checkout", r"check out",
        r"place (?:the |your |my |an )?order",
        r"order now", r"subscrib\w*", r"unsubscrib\w*",
        r"send\w*", r"submit\w*", r"post(?:ing)?\b", r"publish\w*", r"share (?:it )?(?:with|publicly)",
        r"empty(?:ing)? (?:the )?(?:trash|recycle bin|bin)\b", r"overwrit\w*", r"replac\w* (?:the |this |existing |that )?file",
        r"discard\w*", r"don['’]?t save", r"without saving", r"permanently", r"factory reset", r"reset (?:this |the )?(?:pc|computer|phone|settings)",
        r"clear (?:all |my |the )?(?:browsing )?(?:history|data|cookies)",
        r"shut(?:ting)? down", r"restart\w*", r"sign(?:ing)? out", r"log(?:ging)? out",
        r"password", r"passcode", r"credit card", r"bank", r"transfer\w*",
        r"firewall", r"antivirus", r"defender", r"two-factor", r"2fa", r"security settings?", r"privacy settings?",
    ],
    "spanish": [
        r"borr\w*", r"elimin\w*", r"vaci(?:ar|o|a|ando)\b", r"desinstal\w*", r"instal\w*",
        r"compr(?:ar|a|o|e|amos|ando)\b", r"pag(?:ar|o|ue|amos|ando)\b", r"env[ií](?:ar|o|a|e|ando)\b", r"public(?:ar|o|a|ando)\b",
        r"vend(?:er|o|e|emos|iendo)\b", r"transfer\w*", r"contraseña", r"tarjeta de cr[eé]dito",
        r"formate\w* (?:el |la |un |una )?(?:disco|unidad|memoria|partici[oó]n|tarjeta)", r"reinici\w*",
        r"restablec\w*", r"apag(?:ar|o|a|ando)\b", r"cerr(?:ar|o) (?:la )?sesi[oó]n", r"suscrib\w*", r"sobrescrib\w*",
        r"descart\w*", r"no guardar", r"pedido",
    ],
    "french": [
        r"supprim\w*", r"effac\w*", r"vider la corbeille", r"d[ée]sinstall\w*", r"install\w*", r"achet\w*", r"ach[eè]te\w*",
        r"pay(?:er|ez|ons|ant|[ée])\b", r"paiement", r"envoy\w*", r"envoi\w*", r"publi(?:er|e|ez|ons)\b", r"transf[ée]r\w*",
        r"vend(?:re|s|ez|ons)\b", r"mot de passe", r"carte (?:de cr[ée]dit|bancaire)",
        r"formater (?:le |la |un |une )?(?:disque|cl[ée]|partition|carte)", r"r[ée]initialis\w*", r"red[ée]marr\w*",
        r"[ée]teindre", r"d[ée]connect\w*", r"passer (?:la |une |ma )?commande", r"valider (?:la |ma )?commande",
        r"commander\b", r"abonn\w*", r"[ée]cras\w*", r"remplac\w* le fichier", r"ne pas enregistrer",
    ],
    "german": [
        r"l[öo]sch\w*", r"entfern\w*", r"papierkorb leeren", r"leer\w* (?:den )?papierkorb", r"deinstall\w*", r"install\w*",
        r"kauf\w*", r"verkauf\w*", r"bezahl\w*", r"zahlung\w*", r"jetzt zahlen", r"senden", r"versend\w*", r"abschick\w*",
        r"absend\w*", r"ver[öo]ffentlich\w*", r"[üu]berweis\w*", r"passwort", r"kennwort", r"kreditkarte",
        r"formatier\w* (?:die |das |den )?(?:festplatte|laufwerk|partition|sd-karte|usb)", r"zur[üu]cksetz\w*",
        r"neu ?start\w*", r"herunterfahren", r"abmeld\w*", r"bestell\w*", r"abonnier\w*", r"[üu]berschreib\w*",
        r"verwerf\w*", r"nicht speichern",
    ],
    "portuguese": [
        r"apag(?:ar|o|a|ue|ando)\b", r"exclu(?:ir|o|i|a|indo)\b", r"esvazi\w*", r"desinstal\w*", r"instal\w*",
        r"compr(?:ar|o|a|e|ando|ou)\b", r"pag(?:ar|o|ue|amento)\b", r"envi(?:ar|o|a|e|ando)\b", r"public(?:ar|o|a|ando)\b",
        r"vend(?:er|o|e|emos|endo)\b", r"transfer\w*", r"senha", r"cart[ãa]o de cr[ée]dito",
        r"format(?:ar|o|a|ando) (?:o |a |um |uma )?(?:disco|unidade|pen ?drive|cart[ãa]o|parti[çc][ãa]o)",
        r"redefin\w*", r"restaurar o (?:pc|computador)", r"reinici\w*", r"deslig\w*", r"sair da conta",
        r"assin(?:ar|o|e|atura)\b", r"substitu\w* (?:o )?arquivo", r"descart\w*", r"n[ãa]o salvar",
    ],
    "italian": [
        r"elimin\w*", r"cancell\w*", r"svuot\w*", r"disinstall\w*", r"install\w*", r"acquist\w*",
        r"compr(?:are|o|a|iamo)\b", r"vend(?:ere|o|i|iamo)\b", r"pag(?:are|o|a|amento)\b",
        r"invi(?:are|a|ato|iamo)\b", r"pubblic\w*", r"trasferi\w*", r"bonifico", r"carta di credito",
        r"formatt\w* (?:il |l'|un |una )?(?:disco|unit[àa]|chiavetta|scheda|partizione)", r"reimposta\w*",
        r"riavvi\w*", r"spegn\w*", r"disconnett\w*", r"abbon\w*", r"sovrascriv\w*", r"sostitu\w* (?:il )?file",
        r"non salvare",
    ],
}

# Language names (English and native) the check covers.
CHECKED_LANGUAGES = {
    "english", "en", "spanish", "español", "espanol", "es", "french", "français", "francais", "fr",
    "german", "deutsch", "de", "portuguese", "português", "portugues", "pt", "italian", "italiano", "it",
}

# Thoughts are checked against every language: a model asked for Spanish may still write English.
RISKY_INTENT = re.compile(r"\b(" + "|".join(p for ps in _PATTERNS.values() for p in ps) + r")", re.IGNORECASE)


def risky_intent(thought: str) -> str | None:
    """The first hard-to-undo word in a Thought, or None."""
    m = RISKY_INTENT.search(thought or "")
    return m.group(0).lower() if m else None


def language_checked(language: str) -> bool:
    return (language or "english").strip().lower() in CHECKED_LANGUAGES
