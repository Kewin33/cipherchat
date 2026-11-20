"""Synthetic German mini-chat dataset for the CipherChat MVP.

Short prompts, short answers (1-3 words), a mix of
  * memorised small talk / facts (greetings, capitals, colours, opposites, ...)
  * "computed" answers (single-digit +/-, parity, comparisons, counting)
so the encrypted model has something meaningful to do.

Part of the arithmetic/comparison pairs is held out to measure generalisation.
"""
import random

from .vocab import BOS, EOS, SEP, encode_text, normalize

MAX_LEN = 64  # == FHE context window (number of KV-cache blocks)

SMALLTALK = {
    # greetings
    "hallo": "hallo!", "hi": "hi!", "hey": "hey!", "servus": "servus!", "moin": "moin!",
    "moin moin": "moin!", "guten tag": "guten tag!", "guten morgen": "morgen!",
    "guten abend": "guten abend!", "gute nacht": "gute nacht!", "grüß dich": "hallo!",
    "hallöchen": "hallo!", "na du": "na?",
    # how are you
    "wie geht's": "gut, danke!", "wie geht es dir": "gut, danke!", "wie gehts": "gut, danke!",
    "alles gut": "ja, alles gut!", "alles klar": "ja, klar!", "wie läuft's": "läuft!",
    # identity
    "wer bist du": "cipherchat.", "was bist du": "ein fhe-bot.", "wie heißt du": "cipherchat.",
    "wie heisst du": "cipherchat.", "dein name": "cipherchat.", "bist du ein mensch": "nein.",
    "bist du eine ki": "ja.", "bist du ein bot": "ja.", "was kannst du": "rechnen und reden.",
    # privacy
    "siehst du meine daten": "nein, verschlüsselt.", "kannst du mich lesen": "nein.",
    "ist das privat": "ja, dank fhe.", "ist das sicher": "ja, dank fhe.",
    "was ist fhe": "rechnen auf geheimtext.", "wer sieht meine nachricht": "nur du.",
    "was siehst du": "nur zahlensalat.", "kennst du mein passwort": "nein.",
    # thanks / bye
    "danke": "gern!", "danke schön": "gern!", "vielen dank": "gern!", "merci": "gern!",
    "tschüss": "tschüss!", "bye": "bye!", "ciao": "ciao!", "bis dann": "bis dann!",
    "auf wiedersehen": "tschüss!",
    # moods
    "ich bin müde": "schlaf gut!", "ich bin traurig": "kopf hoch!", "ich bin froh": "super!",
    "ich bin glücklich": "super!", "mir ist langweilig": "rechne was!", "ich habe hunger": "iss was!",
    "erzähl einen witz": "fhe ist schnell.", "ja": "gut.", "nein": "schade.", "ok": "ok!",
    "magst du mathe": "ja!", "wie spät ist es": "keine ahnung.", "wo wohnst du": "im server.",
}

CAPITALS = {
    "frankreich": "paris", "deutschland": "berlin", "italien": "rom", "spanien": "madrid",
    "österreich": "wien", "schweiz": "bern", "polen": "warschau", "england": "london",
    "japan": "tokio", "china": "peking", "russland": "moskau", "griechenland": "athen",
    "portugal": "lissabon", "belgien": "brüssel", "schweden": "stockholm", "norwegen": "oslo",
    "dänemark": "kopenhagen", "türkei": "ankara", "ägypten": "kairo", "kanada": "ottawa",
    "irland": "dublin", "finnland": "helsinki", "ungarn": "budapest", "tschechien": "prag",
}

COLOURS = {
    "gras": "grün", "dem himmel": "blau", "schnee": "weiß", "blut": "rot", "einer banane": "gelb",
    "kohle": "schwarz", "einer tomate": "rot", "milch": "weiß", "dem meer": "blau",
    "einer zitrone": "gelb", "einem frosch": "grün", "der sonne": "gelb",
}

OPPOSITES = {
    "groß": "klein", "klein": "groß", "heiß": "kalt", "kalt": "heiß", "hell": "dunkel",
    "dunkel": "hell", "schnell": "langsam", "langsam": "schnell", "oben": "unten", "unten": "oben",
    "links": "rechts", "rechts": "links", "tag": "nacht", "nacht": "tag", "laut": "leise",
    "leise": "laut", "voll": "leer", "leer": "voll", "lang": "kurz", "kurz": "lang",
    "stark": "schwach", "schwach": "stark", "jung": "alt", "alt": "jung", "gut": "schlecht",
    "schlecht": "gut", "nass": "trocken", "trocken": "nass", "hoch": "tief", "tief": "hoch",
}

WEEKDAYS = ["montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag"]

PREFIXES = ["", "", "", "", "hey, ", "sag mal, ", "bitte, ", "kurze frage: "]


def _q(p, rng, punct="?"):
    """Random light augmentation of a prompt."""
    pre = rng.choice(PREFIXES)
    suf = rng.choice([punct, punct, "", "!" if punct == "?" else "?"])
    return normalize(pre + p + suf)


def build_pairs(seed: int = 0, holdout_frac: float = 0.15):
    """Return (train_items, test_items); each item = (category, base_prompt_fn, answer)."""
    rng = random.Random(seed)
    train, test = [], []

    for p, a in SMALLTALK.items():
        train.append(("smalltalk", p, a))
    for c, cap in CAPITALS.items():
        train.append(("hauptstadt", f"hauptstadt von {c}", cap + "."))
        train.append(("hauptstadt", f"was ist die hauptstadt von {c}", cap + "."))
    for thing, col in COLOURS.items():
        train.append(("farbe", f"welche farbe hat {thing}", col + "."))
    for w, o in OPPOSITES.items():
        train.append(("gegenteil", f"gegenteil von {w}", o + "."))
        train.append(("gegenteil", f"was ist das gegenteil von {w}", o + "."))
    for i, d in enumerate(WEEKDAYS):
        train.append(("wochentag", f"was kommt nach {d}", WEEKDAYS[(i + 1) % 7] + "."))
    for n in range(1, 6):
        train.append(("zählen", f"zähl bis {n}", " ".join(str(k) for k in range(1, n + 1))))
    for n in range(10):
        train.append(("parität", f"ist {n} gerade", "ja." if n % 2 == 0 else "nein."))
        train.append(("parität", f"ist {n} ungerade", "nein." if n % 2 == 0 else "ja."))

    # arithmetic & comparisons: hold out some (a, b) pairs for generalisation tests
    arith = []
    for a in range(10):
        for b in range(10):
            arith.append(("plus", a, b))
            if a >= b:
                arith.append(("minus", a, b))
            if a != b:
                arith.append(("größer", a, b))
    rng.shuffle(arith)
    n_hold = int(len(arith) * holdout_frac)
    for k, (op, a, b) in enumerate(arith):
        if op == "plus":
            items = [("rechnen", f"was ist {a}+{b}", str(a + b)), ("rechnen", f"{a}+{b}", str(a + b)),
                     ("rechnen", f"{a} plus {b}", str(a + b))]
        elif op == "minus":
            items = [("rechnen", f"was ist {a}-{b}", str(a - b)), ("rechnen", f"{a}-{b}", str(a - b)),
                     ("rechnen", f"{a} minus {b}", str(a - b))]
        else:
            items = [("vergleich", f"ist {a} größer als {b}", "ja." if a > b else "nein."),
                     ("vergleich", f"ist {a} kleiner als {b}", "ja." if a < b else "nein.")]
        (test if k < n_hold else train).extend(items)
    return train, test


def to_ids(prompt: str, answer: str):
    ids = [BOS] + encode_text(prompt) + [SEP] + encode_text(answer) + [EOS]
    return ids


def sample_batch(items, batch_size, rng, max_len=MAX_LEN):
    """Return (ids, loss_mask) as python lists, padded to the longest sequence."""
    seqs, masks = [], []
    while len(seqs) < batch_size:
        cat, p, a = rng.choice(items)
        punct = "?" if cat not in ("zählen",) else ""
        prompt = _q(p, rng, punct) if cat != "rechnen" or rng.random() < 0.5 else normalize(p + rng.choice(["", "?", "=", " =?"]))
        ids = to_ids(prompt, a)
        if len(ids) > max_len:
            continue
        sep_pos = ids.index(SEP)
        m = [0] * len(ids)
        for i in range(sep_pos, len(ids) - 1):  # predict answer tokens + <eos>
            m[i] = 1
        seqs.append(ids)
        masks.append(m)
    T = max(len(s) for s in seqs)
    ids = [s + [0] * (T - len(s)) for s in seqs]
    msk = [m + [0] * (T - len(m)) for m in masks]
    return ids, msk


def eval_prompts(items):
    """Canonical (un-augmented) prompts for exact-match evaluation."""
    out = []
    for cat, p, a in items:
        out.append((cat, normalize(p + ("?" if cat not in ("zählen", "rechnen") else "")), a))
    return out
