"""Stronger ciphertext-only attacks on the two *learnable* AES variants of the experiment.

Threat model: the attacker (= the server that hosts the model) only sees ciphertexts, but has
public text of the same kind (here: 2,000 *different* chat messages from the same distribution).

  C) AES-ECB per character  -> a substitution cipher: unigram ranks + bigram hill climbing
  D) AES-CTR with fixed nonce -> a many-time pad: per-position keystream byte by likelihood
"""
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aes_learnability as X  # noqa: E402  (same corpus generator + encoders, fresh random key)
from Crypto.Cipher import AES  # noqa: E402

ref_text = " ".join(X.msgs_val)            # attacker's public corpus (disjoint messages)
victim = X.msgs_train[:5000]                # intercepted traffic


def attack_ecb(victim_msgs):
    ct = [X.enc_ecb_char(m) for m in victim_msgs]
    true_map = {X.BLOCK_IDS[b]: X.ECB.decrypt(b).rstrip(b"\0").decode("utf-8") for b in X.BLOCK_IDS}
    alphabet = sorted(set(ref_text) | set(true_map.values()))
    ai = {c: i for i, c in enumerate(alphabet)}
    m = len(alphabet)
    # bigram log-probabilities from the public corpus
    big = np.ones((m, m))
    for a, b in zip(ref_text, ref_text[1:]):
        big[ai[a], ai[b]] += 1
    L = np.log(big / big.sum(1, keepdims=True))
    # ciphertext statistics
    toks = sorted({t for s in ct for t in s})
    ti = {t: i for i, t in enumerate(toks)}
    n = len(toks)
    C = np.zeros((n, n))
    for s in ct:
        for a, b in zip(s, s[1:]):
            C[ti[a], ti[b]] += 1
    # init: unigram rank matching
    ref_rank = [ai[c] for c, _ in Counter(ref_text).most_common()]
    ref_rank += [i for i in range(m) if i not in ref_rank]
    ct_rank = [ti[t] for t, _ in Counter(t for s in ct for t in s).most_common()]
    pi = np.zeros(n, dtype=int)
    used = set()
    for r, t in enumerate(ct_rank):
        pi[t] = ref_rank[r]
        used.add(ref_rank[r])
    free = [i for i in range(m) if i not in used]

    def acc(p):
        tot = ok = 0
        for s in ct:
            for t in s:
                tot += 1
                ok += alphabet[p[ti[t]]] == true_map[t]
        return ok / tot

    acc_unigram = acc(pi)
    score = lambda p: float((C * L[np.ix_(p, p)]).sum())
    best = score(pi)
    improved = True
    while improved:
        improved = False
        for a in range(n):
            for b in range(a + 1, n):
                pi[a], pi[b] = pi[b], pi[a]
                s = score(pi)
                if s > best + 1e-9:
                    best, improved = s, True
                else:
                    pi[a], pi[b] = pi[b], pi[a]
            for j, f in enumerate(free):            # also try unused plaintext symbols
                old = pi[a]
                pi[a] = f
                s = score(pi)
                if s > best + 1e-9:
                    best, improved = s, True
                    free[j] = old
                else:
                    pi[a] = old
    return acc_unigram, acc(pi)


def attack_ctr_fixed(victim_msgs):
    ct = [X.enc_ctr_fixed(m) for m in victim_msgs]
    freq = Counter(ref_text.encode("utf-8"))
    tot_f = sum(freq.values())
    logp = np.array([math.log((freq.get(b, 0) + 0.01) / (tot_f + 2.56)) for b in range(256)])
    T = max(len(s) for s in ct)
    true_ks = AES.new(X.KEY, AES.MODE_CTR, nonce=X.FIXED_NONCE).encrypt(bytes(T))
    ks = []
    for i in range(T):
        col = np.array([s[i] for s in ct if len(s) > i])
        scores = [logp[col ^ k].sum() for k in range(256)]
        ks.append(int(np.argmax(scores)))
    tot = ok = 0
    for s in ct:
        for i, c in enumerate(s):
            tot += 1
            ok += ks[i] == true_ks[i]
    return ok / tot


if __name__ == "__main__":
    u, h = attack_ecb(victim)
    d = attack_ctr_fixed(victim)
    print(f"C) AES-ECB pro Zeichen: Unigramm-Häufigkeiten {u*100:.1f}% -> + Bigramm-Hill-Climbing {h*100:.1f}% der Zeichen entschlüsselt")
    print(f"D) AES-CTR feste Nonce: {d*100:.1f}% der Bytes entschlüsselt (nur Ciphertext)")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "aes_experiment_results.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    res.update(attack_ecb_unigram_acc=u, attack_ecb_bigram_hillclimb_acc=h, attack_ctr_fixed_nonce_same_domain_acc=d)
    json.dump(res, open(path, "w"), indent=2, ensure_ascii=False)
