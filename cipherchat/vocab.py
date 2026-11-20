"""Character-level vocabulary for CipherChat.

The vocabulary must fit into one FHE block (B = 64 slots), so we keep it tiny:
lowercase German letters, digits, a few punctuation marks and 5 special tokens.
"""

SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>", "<unk>"]
CHARS = "abcdefghijklmnopqrstuvwxyzäöüß0123456789 .,?!-+=:'()*/"

ITOS = SPECIALS + list(CHARS)
STOI = {c: i for i, c in enumerate(ITOS)}
PAD, BOS, SEP, EOS, UNK = range(5)
VOCAB_SIZE = len(ITOS)
assert VOCAB_SIZE <= 64, VOCAB_SIZE


def normalize(text: str) -> str:
    """Lowercase and map a few common characters onto the vocabulary."""
    t = text.lower().strip()
    t = t.replace("ẞ", "ß").replace("’", "'").replace("„", "").replace("“", "")
    t = " ".join(t.split())  # collapse whitespace
    return t


def encode_text(text: str) -> list[int]:
    return [STOI.get(c, UNK) for c in normalize(text)]


def encode_prompt(prompt: str) -> list[int]:
    """<bos> prompt <sep>  -> the model then generates the answer until <eos>."""
    return [BOS] + encode_text(prompt) + [SEP]


def decode(ids) -> str:
    out = []
    for i in ids:
        if i in (PAD, BOS, SEP, EOS):
            continue
        out.append(ITOS[i] if i != UNK else "?")
    return "".join(out)
