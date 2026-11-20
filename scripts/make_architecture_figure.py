"""Erzeugt die Architektur-Grafik von CipherFormer2 (cipherchat/v2/model.py) als SVG.

Aufruf:  python scripts/make_architecture_figure.py [ausgabe.svg]
Standard: docs/architecture/cipherformer2.svg   (PNG: siehe Hinweis am Ende)
Nur Standardbibliothek, keine Abhängigkeiten.
"""
import os
import sys
from xml.sax.saxutils import escape

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "docs", "architecture", "cipherformer2.svg")

W, H = 1120, 960
parts = []


def add(s):
    parts.append(s)


def text(x, y, s, size=13, anchor="start", weight="normal", fill="#1f2937", family="DejaVu Sans"):
    add(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
        f'fill="{fill}" font-family="{family}">{escape(s)}</text>')


def box(x, y, w, h, title, lines=(), fill="#eef4ff", stroke="#3b6fd8", dashed=False):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="1.6"{dash}/>')
    text(x + w / 2, y + 22, title, size=14, anchor="middle", weight="bold")
    for i, ln in enumerate(lines):
        text(x + w / 2, y + 44 + i * 19, ln, size=12.5, anchor="middle", fill="#374151")


def arrow(x1, y1, x2, y2, color="#4b5563"):
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6" '
        f'marker-end="url(#ah)"/>')


def line(x1, y1, x2, y2, color="#9ca3af", dashed=False):
    dash = ' stroke-dasharray="4 4"' if dashed else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.4"{dash}/>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#4b5563"/></marker></defs>')
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

text(40, 42, "CipherFormer2 – Architektur (v2, FHE-kompatibel)", size=22, weight="bold")
text(40, 66, "Dekoder-only Transformer, 2 Layer, 1 Head, d = 64, kein LayerNorm, kein Softmax",
     size=13, fill="#6b7280")

CX = 430  # center column of the flow

# input
box(CX - 230, 90, 460, 56, "Eingabe: Token-ID → One-Hot",
    ["Vokabular 59 + konstanter Kanal x[63] = 1 (trägt alle Biases)"], fill="#f3f4f6", stroke="#6b7280")
arrow(CX, 146, CX, 180)

# embeddings
box(CX - 230, 180, 220, 58, "E_attn (d = 64)", ["Eingang der Attention"], fill="#fef3c7", stroke="#b45309")
box(CX + 10, 180, 220, 58, "E_res (d = 64)", ["Residualstrom"], fill="#fef3c7", stroke="#b45309")
arrow(CX - 120, 238, CX - 120, 262)
arrow(CX + 120, 238, CX + 120, 262)


def block(y0, title, mlp):
    """Paralleler Block: x ← x + α_a·Attn + α_m·MLP. Attention und MLP lesen dieselbe Eingabe."""
    bh = 232
    add(f'<rect x="{CX - 300}" y="{y0}" width="600" height="{bh}" rx="12" fill="none" '
        f'stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="6 4"/>')
    text(CX - 290, y0 + 18, title, size=12, fill="#6b7280", weight="bold")
    # attention
    ax, aw = CX - 270, 250
    box(ax, y0 + 30, aw, 150, "Attention (1 Head, kausal)", [
        "q = x·Wq,  k = x·Wk,  v = x·Wv",
        "s_tj = q_t·k_j + c + ρ[t−j]",
        "w_tj = κ · s_tj²",
        "out = Σ_j w_tj · v_j",
    ], fill="#eef4ff", stroke="#3b6fd8")
    # mlp
    mx, mw = CX + 20, 250
    if mlp:
        box(mx, y0 + 30, mw, 150, "MLP (gated, polynomial)", [
            "z = x·W1,  g = x·Wg",
            "m = (z ⊙ g)²",
            "out = m·W2 + b2",
            "Breite d_ff = 1024 bis 2048",
        ], fill="#ecfdf5", stroke="#047857")
    else:
        box(mx, y0 + 30, mw, 150, "kein MLP (Layer 0)", [
            "Bei One-Hot-Eingabe ist ein MLP",
            "nur eine Tabelle: E_attn und E_res",
            "übernehmen diese Rolle.",
        ], fill="#f9fafb", stroke="#9ca3af", dashed=True)
    # residual stream on the left, branches down to the sum node
    sy = y0 + 212
    line(CX - 300, y0 + 60, CX - 300, sy, color="#b45309")
    add(f'<line x1="{CX - 300}" y1="{sy}" x2="{CX - 26}" y2="{sy}" stroke="#b45309" stroke-width="1.6" '
        f'marker-end="url(#ah)"/>')
    line(ax + aw / 2, y0 + 180, ax + aw / 2, sy - 2)
    line(mx + mw / 2, y0 + 180, mx + mw / 2, sy - 2)
    line(ax + aw / 2, sy - 2, CX - 14, sy - 2)
    line(mx + mw / 2, sy - 2, CX + 14, sy - 2)
    add(f'<circle cx="{CX}" cy="{sy}" r="12" fill="#ffffff" stroke="#374151" stroke-width="1.6"/>')
    text(CX, sy + 5, "+", size=16, anchor="middle", weight="bold")
    text(CX + 22, sy + 4, "x ← x + α_a·Attn + α_m·MLP", size=12, fill="#374151")
    return y0 + bh


end0 = block(262, "Layer 0 (Lookup-Schicht)", mlp=False)
arrow(CX, end0, CX, end0 + 22)
end1 = block(end0 + 22, "Layer 1 (letzter Layer bei L = 2)", mlp=True)
arrow(CX, end1, CX, end1 + 22)

# unembedding and output
uy = end1 + 22
box(CX - 230, uy, 460, 50, "Unembedding", ["logits = x·Wu + b_u   (V = 59)"], fill="#f5f3ff", stroke="#6d28d9")
arrow(CX, uy + 50, CX, uy + 74)
box(CX - 230, uy + 74, 460, 50, "Greedy-Decoding", ["nächstes Token = argmax(logits)"],
    fill="#f3f4f6", stroke="#6b7280")

# side notes
NX = 790
text(NX, 300, "Besonderheiten für FHE", size=15, weight="bold")
notes = [
    "• Keine LayerNorm, kein Softmax:",
    "   nur Polynome (Grad 2 bzw. 4)",
    "• Paralleler Block: Attention und",
    "   MLP lesen dieselbe Eingabe",
    "• Relativer Positions-Bias ρ[t−j]",
    "   statt absoluter Positionen",
    "• Feste Normierung κ (statt 1/(t+1))",
    "• Bilineare Scores: der Cache speichert",
    "   k̃_j = x_j·Mᵀ, keine q-Projektion",
    "• Layer 0 ohne MLP: Embedding-Lookup",
    "• Schiebender KV-Cache: pro Token um",
    "   einen 128-Slot-Block (statische Plaintexts)",
]
for i, ln in enumerate(notes):
    text(NX, 326 + i * 20, ln, size=12.5, fill="#374151")

box(NX - 10, 600, 300, 120, "Kennzahlen (Forschungsmodell)", [
    "d = 64, L = 2, 1 Head",
    "Kontext ≤ 64 Tokens",
    "Vokabular 59",
    "Additionsmodell: 428 k Parameter",
], fill="#fffbeb", stroke="#b45309")

text(NX, 760, "Legende", size=13, weight="bold")
add(f'<rect x="{NX}" y="772" width="16" height="12" fill="#eef4ff" stroke="#3b6fd8"/>')
text(NX + 24, 783, "Attention", size=12, fill="#374151")
add(f'<rect x="{NX}" y="792" width="16" height="12" fill="#ecfdf5" stroke="#047857"/>')
text(NX + 24, 803, "MLP", size=12, fill="#374151")
add(f'<rect x="{NX}" y="812" width="16" height="12" fill="#fef3c7" stroke="#b45309"/>')
text(NX + 24, 823, "Embedding-Tabellen", size=12, fill="#374151")

text(40, H - 22, "Quelle: cipherchat/v2/model.py (CipherFormer2), erzeugt mit scripts/make_architecture_figure.py",
     size=11, fill="#9ca3af")

add("</svg>")

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(parts))
print("geschrieben:", os.path.abspath(OUT))
