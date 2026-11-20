"""Dataset v2: the v1 mini-chat categories + multi-digit addition with HELD-OUT operand pairs.

Generalisation protocol: all unordered pairs {a, b} (0 <= a, b <= max_n) are split once (fixed seed);
a held-out pair never appears in training in either order or any template. Exact match on the held-out
pairs therefore measures whether the model learned addition rather than memorising the table.

Number format option `rev` (little-endian digits, "68" is written "86"): applied by the CLIENT to every
digit run of the prompt and undone on the answer, i.e. a tokenization choice, invisible to the user.
"""
import random

from ..dataset import CAPITALS, COLOURS, OPPOSITES, PREFIXES, SMALLTALK, WEEKDAYS
from ..vocab import BOS, EOS, SEP, STOI, VOCAB_SIZE, encode_text, normalize

MAX_LEN = 64
DIGITS = {STOI[str(d)] for d in range(10)}
N_SIG = 64 - VOCAB_SIZE          # free one-hot dimensions 59..63 carry the digit significance (1..5)


def significance(ids, rev=True):
    """Client-side position coupling (Cho et al. 2024): tag every token with the significance of a digit.
    Prompt digits get their own significance (1 = units); <sep> gets 1 (the next token to produce is the
    units digit); a generated digit that is the k-th digit of its run gets k+1 (significance of the NEXT
    digit). Everything else gets 0. Only defined by the token ids -> recomputable at every decoding step.
    With rev=False the prompt digits are counted from the end of their run (big-endian)."""
    sig = [0] * len(ids)
    try:
        sep = ids.index(SEP)
    except ValueError:
        sep = len(ids)
    i = 0
    while i < sep:                                   # prompt part
        if ids[i] in DIGITS:
            j = i
            while j < sep and ids[j] in DIGITS:
                j += 1
            n = j - i
            for k in range(n):
                sig[i + k] = min(N_SIG, (k + 1) if rev else (n - k))
            i = j
        else:
            i += 1
    if sep < len(ids):
        sig[sep] = 1
        run = 0
        for i in range(sep + 1, len(ids)):           # answer part (generated)
            if ids[i] in DIGITS:
                run += 1
                sig[i] = min(N_SIG, run + 1)
            else:
                run = 0
    return sig

ADD_TEMPLATES = ["was ist {a}+{b}", "{a}+{b}", "{a} plus {b}", "was ist {a} plus {b}", "wie viel ist {a}+{b}",
                 "rechne {a}+{b}", "{a} + {b}", "was ergibt {a}+{b}"]
ADD_SUFFIXES = ["?", "?", "", "=", " =", "=?", " =?"]
CANONICAL = "was ist {a}+{b}?"


def reverse_digits(text: str) -> str:
    """Reverse every maximal run of digits ("was ist 23+45?" -> "was ist 32+54?")."""
    out, run = [], []
    for ch in text:
        if ch.isdigit():
            run.append(ch)
            continue
        out.extend(reversed(run))
        run = []
        out.append(ch)
    out.extend(reversed(run))
    return "".join(out)


def split_pairs(max_n=99, test_frac=0.2, seed=0):
    rng = random.Random(seed)
    pairs = [(a, b) for a in range(max_n + 1) for b in range(a, max_n + 1)]
    rng.shuffle(pairs)
    n_test = int(len(pairs) * test_frac)
    return sorted(pairs[n_test:]), sorted(pairs[:n_test])


def other_items():
    """(category, prompt, answer) for the non-addition chat categories (all used for training)."""
    items = []
    for p, a in SMALLTALK.items():
        items.append(("smalltalk", p, a))
    for c, cap in CAPITALS.items():
        items.append(("hauptstadt", f"hauptstadt von {c}", cap + "."))
        items.append(("hauptstadt", f"was ist die hauptstadt von {c}", cap + "."))
    for thing, col in COLOURS.items():
        items.append(("farbe", f"welche farbe hat {thing}", col + "."))
    for w, o in OPPOSITES.items():
        items.append(("gegenteil", f"gegenteil von {w}", o + "."))
        items.append(("gegenteil", f"was ist das gegenteil von {w}", o + "."))
    for i, d in enumerate(WEEKDAYS):
        items.append(("wochentag", f"was kommt nach {d}", WEEKDAYS[(i + 1) % 7] + "."))
    for n in range(1, 6):
        items.append(("zählen", f"zähl bis {n}", " ".join(str(k) for k in range(1, n + 1))))
    for n in range(10):
        items.append(("parität", f"ist {n} gerade", "ja." if n % 2 == 0 else "nein."))
        items.append(("parität", f"ist {n} ungerade", "nein." if n % 2 == 0 else "ja."))
    for a in range(10):
        for b in range(10):
            if a >= b:
                items.append(("minus", f"was ist {a}-{b}", str(a - b)))
            if a != b:
                items.append(("vergleich", f"ist {a} größer als {b}", "ja." if a > b else "nein."))
                items.append(("vergleich", f"ist {a} kleiner als {b}", "ja." if a < b else "nein."))
    return items


class Task:
    def __init__(self, max_n=99, test_frac=0.2, seed=0, rev=False, p_add=0.5, coupling=False):
        self.max_n, self.rev, self.p_add, self.coupling = max_n, rev, p_add, coupling
        self.train_pairs, self.test_pairs = split_pairs(max_n, test_frac, seed)
        self.others = other_items()

    def fmt(self, text):
        return reverse_digits(text) if self.rev else text

    def to_ids(self, prompt, answer):
        return [BOS] + encode_text(self.fmt(prompt)) + [SEP] + encode_text(self.fmt(answer)) + [EOS]

    def sample(self, rng):
        if rng.random() < self.p_add:
            a, b = rng.choice(self.train_pairs)
            if rng.random() < 0.5:
                a, b = b, a
            p = rng.choice(ADD_TEMPLATES).format(a=a, b=b) + rng.choice(ADD_SUFFIXES)
            if rng.random() < 0.2:
                p = rng.choice(PREFIXES) + p
            return normalize(p), str(a + b)
        cat, p, ans = rng.choice(self.others)
        punct = "" if cat == "zählen" else "?"
        p = rng.choice(PREFIXES) + p + rng.choice([punct, punct, "", "!" if punct == "?" else "?"])
        return normalize(p), ans

    def sig(self, ids):
        return significance(ids, self.rev) if self.coupling else [0] * len(ids)

    def batch(self, n, rng):
        seqs, masks, sigs = [], [], []
        while len(seqs) < n:
            p, a = self.sample(rng)
            ids = self.to_ids(p, a)
            if len(ids) > MAX_LEN:
                continue
            sep = ids.index(SEP)
            m = [0] * len(ids)
            for i in range(sep, len(ids) - 1):
                m[i] = 1
            seqs.append(ids)
            masks.append(m)
            sigs.append(self.sig(ids))
        T = max(len(s) for s in seqs)
        pad = lambda rows: [r + [0] * (T - len(r)) for r in rows]
        return pad(seqs), pad(masks), pad(sigs)

    # ---- evaluation sets: (category, prompt, answer) in user-facing (big-endian) format ----
    def eval_addition(self, pairs, template=CANONICAL, both_orders=True):
        out = []
        for a, b in pairs:
            out.append(("add", template.format(a=a, b=b), str(a + b)))
            if both_orders and a != b:
                out.append(("add", template.format(a=b, b=a), str(a + b)))
        return out

    def eval_others(self):
        out = []
        for cat, p, a in self.others:
            out.append((cat, normalize(p + ("" if cat == "zählen" else "?")), a))
        return out

    def prompt_ids(self, prompt):
        return [BOS] + encode_text(self.fmt(normalize(prompt))) + [SEP]

    def decode_answer(self, text):
        return reverse_digits(text) if self.rev else text
