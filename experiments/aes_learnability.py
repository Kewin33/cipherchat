"""Experiment for approach 2: "supervised learning on AES ciphertext -> next ciphertext".

Can a neural network learn next-token prediction directly on AES-encrypted text?
We train the same small model on four encodings of the same German chat corpus:

  A) plaintext bytes                              (baseline)
  B) AES-128-CTR, fresh random nonce per message  (correct, IND-CPA secure usage)
  C) AES-128-ECB per character, fixed key         (deterministic "token encryption")
  D) AES-128-CTR with a FIXED nonce               (nonce reuse)

and then attack the variants that turn out to be learnable.
Run: python experiments/aes_learnability.py   (CPU, ~10 min)
"""
import json
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cipherchat.dataset import _q, build_pairs  # noqa: E402

torch.manual_seed(0)
random.seed(0)
rng = random.Random(0)
KEY = get_random_bytes(16)

# ---------------- corpus ----------------
train_items, test_items = build_pairs(0)
items = train_items + test_items


def make_messages(n):
    out = []
    for _ in range(n):
        cat, p, a = rng.choice(items)
        out.append((_q(p, rng) + " " + a))
    return out


msgs_train = make_messages(20000)
msgs_val = make_messages(2000)
MAXLEN = 64


# ---------------- encodings ----------------
def enc_plain(m):
    return list(m.encode("utf-8"))[:MAXLEN]


def enc_ctr_random(m):
    nonce = get_random_bytes(8)
    c = AES.new(KEY, AES.MODE_CTR, nonce=nonce).encrypt(m.encode("utf-8"))
    return list(c)[:MAXLEN]  # (nonce sent in clear as usual; not given to the model)


FIXED_NONCE = get_random_bytes(8)


def enc_ctr_fixed(m):
    c = AES.new(KEY, AES.MODE_CTR, nonce=FIXED_NONCE).encrypt(m.encode("utf-8"))
    return list(c)[:MAXLEN]


ECB = AES.new(KEY, AES.MODE_ECB)
BLOCK_IDS = {}


def enc_ecb_char(m):
    toks = []
    for ch in m:
        blk = ECB.encrypt(ch.encode("utf-8").ljust(16, b"\0"))
        if blk not in BLOCK_IDS:
            BLOCK_IDS[blk] = len(BLOCK_IDS)
        toks.append(BLOCK_IDS[blk])
    return toks[:MAXLEN]


# ---------------- model ----------------
class LM(nn.Module):
    def __init__(self, vocab, d=128, h=256):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, d)
        self.rnn = nn.GRU(d, h, batch_first=True, num_layers=2)
        self.out = nn.Linear(h, vocab + 1)

    def forward(self, x):
        y, _ = self.rnn(self.emb(x))
        return self.out(y)


def batches(seqs, bs, pad):
    idx = list(range(len(seqs)))
    while True:
        rng.shuffle(idx)
        for i in range(0, len(idx) - bs, bs):
            chunk = [seqs[j] for j in idx[i:i + bs]]
            T = max(len(s) for s in chunk)
            x = torch.full((bs, T), pad, dtype=torch.long)
            for r, s in enumerate(chunk):
                x[r, : len(s)] = torch.tensor(s)
            yield x


def evaluate(model, seqs, pad):
    model.eval()
    tot, n, correct = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, len(seqs), 256):
            chunk = seqs[i:i + 256]
            T = max(len(s) for s in chunk)
            x = torch.full((len(chunk), T), pad, dtype=torch.long)
            for r, s in enumerate(chunk):
                x[r, : len(s)] = torch.tensor(s)
            logits = model(x[:, :-1])
            tgt = x[:, 1:]
            mask = tgt != pad
            lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            tot += -(lp * mask).sum().item()
            n += mask.sum().item()
            correct += ((logits.argmax(-1) == tgt) & mask).sum().item()
    model.train()
    return tot / n, correct / n


def run(name, enc, vocab, steps=1500):
    tr = [enc(m) for m in msgs_train]
    va = [enc(m) for m in msgs_val]
    if vocab is None:
        vocab = len(BLOCK_IDS)
    model = LM(vocab)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    gen = batches(tr, 64, vocab)
    t0 = time.time()
    for step in range(steps):
        x = next(gen)
        logits = model(x[:, :-1])
        tgt = x[:, 1:]
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab + 1), tgt.reshape(-1), ignore_index=vocab)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    ce, acc = evaluate(model, va, vocab)
    chance = math.log(vocab)
    print(f"{name:42s} val CE {ce:.3f} nats (chance {chance:.3f}) | next-token acc {acc*100:5.1f}% | "
          f"{time.time()-t0:.0f}s", flush=True)
    return dict(name=name, val_ce=ce, chance_ce=chance, acc=acc, vocab=vocab), tr


# ---------------- attacks ----------------
GERMAN_FREQ_TEXT = (
    "die sonne scheint heute über der stadt und viele menschen gehen im park spazieren. "
    "kinder spielen auf der wiese, während ihre eltern auf den bänken sitzen und reden. "
    "am abend treffen sich freunde in einem kleinen café, trinken kaffee und erzählen von "
    "ihrer woche. wer möchte, kann noch einen film im kino ansehen oder ein buch lesen. "
    "das wetter soll morgen schlechter werden, deshalb nehmen alle einen schirm mit. "
    "ist das nicht schön? ja, sicher. danke für die frage und bis bald!"
)


def attack_ecb_frequency(tr_tokens, msgs):
    """Ciphertext-only: map ciphertext token ranks to German letter-frequency ranks."""
    ref = [c for c, _ in Counter(GERMAN_FREQ_TEXT).most_common()]
    ct_rank = [t for t, _ in Counter(t for s in tr_tokens for t in s).most_common()]
    guess = {t: (ref[i] if i < len(ref) else "?") for i, t in enumerate(ct_rank)}
    inv = {v: k for k, v in BLOCK_IDS.items()}
    true_map = {BLOCK_IDS[b]: ECB.decrypt(b).rstrip(b"\0").decode("utf-8") for b in BLOCK_IDS}
    tot = sum(1 for s in tr_tokens for _ in s)
    ok = sum(1 for s in tr_tokens for t in s if guess[t] == true_map[t])
    return ok / tot


def attack_ctr_fixed_nonce(ct_seqs, n_msgs=2000):
    """Ciphertext-only many-time-pad attack: per position choose the keystream byte that makes the
    decrypted column look most like German text (unigram log-likelihood)."""
    freq = Counter(GERMAN_FREQ_TEXT.encode("utf-8"))
    logp = {b: math.log((freq.get(b, 0) + 0.1) / (sum(freq.values()) + 25.6)) for b in range(256)}
    seqs = ct_seqs[:n_msgs]
    T = max(len(s) for s in seqs)
    ks = []
    for i in range(T):
        col = [s[i] for s in seqs if len(s) > i]
        best = max(range(256), key=lambda k: sum(logp[c ^ k] for c in col))
        ks.append(best)
    true_ks = AES.new(KEY, AES.MODE_CTR, nonce=FIXED_NONCE).encrypt(bytes(T))
    # fraction of ciphertext bytes that decrypt correctly with the recovered keystream
    tot = ok = 0
    for s in seqs:
        for i, c in enumerate(s):
            tot += 1
            ok += (c ^ ks[i]) == (c ^ true_ks[i])
    return ok / tot


if __name__ == "__main__":
    results = []
    r, _ = run("A) Klartext (UTF-8 Bytes)", enc_plain, 256); results.append(r)
    r, _ = run("B) AES-CTR, zufällige Nonce (korrekt)", enc_ctr_random, 256); results.append(r)
    r, ecb_tr = run("C) AES-ECB pro Zeichen (deterministisch)", enc_ecb_char, None); results.append(r)
    r, ctr_tr = run("D) AES-CTR, feste Nonce (Nonce-Reuse)", enc_ctr_fixed, 256); results.append(r)
    a_ecb = attack_ecb_frequency(ecb_tr, msgs_train)
    a_ctr = attack_ctr_fixed_nonce(ctr_tr)
    print(f"Angriff auf C (nur Ciphertext, Häufigkeitsanalyse): {a_ecb*100:.1f}% der Zeichen korrekt entschlüsselt")
    print(f"Angriff auf D (nur Ciphertext, Many-Time-Pad):     {a_ctr*100:.1f}% der Bytes korrekt entschlüsselt")
    os.makedirs("docs", exist_ok=True)
    with open("docs/aes_experiment_results.json", "w") as f:
        json.dump(dict(results=results, attack_ecb_char_acc=a_ecb, attack_ctr_fixed_nonce_byte_acc=a_ctr), f,
                  indent=2, ensure_ascii=False)
