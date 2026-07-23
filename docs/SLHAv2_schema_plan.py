#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLHA v2 — Schéma de principe + logigramme + plan d'amélioration (PDF vectoriel).

Génère docs/SLHAv2_schema_plan.pdf avec reportlab :
  p1  Vue d'ensemble du projet
  p2  Schéma de principe (data-flow d'encodage / score / tuile / Soft-Paging)
  p3  Logigramme (flow d'exécution : encode, budget, score)
  p4+ Plan d'amélioration (axes priorisés + roadmap + grounding littérature)
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ----------------------------------------------------------------------------
# Fonts: try to register a Unicode TTF so accents + Greek render. Fall back to
# the built-in Helvetica which *is* Latin-1 (covers French accents) — fine.
# ----------------------------------------------------------------------------
def _register_fonts():
    candidates = [
        ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("DejaVuSans-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    registered = {}
    for name, path in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                registered[name] = path
            except Exception:
                pass
    if "DejaVuSans" in registered and "DejaVuSans-Bold" in registered:
        return "DejaVuSans", "DejaVuSans-Bold"
    return "Helvetica", "Helvetica-Bold"

REG, REGB = _register_fonts()

# ----------------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------------
INK      = colors.HexColor("#101828")
SUB      = colors.HexColor("#475467")
ACCENT   = colors.HexColor("#1d4ed8")   # blue
ACCENT2  = colors.HexColor("#0e7490")   # teal
HOT      = colors.HexColor("#b45309")   # amber
WARM     = colors.HexColor("#9a3412")   # orange-red
COLD     = colors.HexColor("#475467")   # grey
LATENT   = colors.HexColor("#1d4ed8")
RESID    = colors.HexColor("#7c3aed")   # purple
META     = colors.HexColor("#0e7490")
GREEN    = colors.HexColor("#15803d")
RED      = colors.HexColor("#b91c1c")
PAPER    = colors.HexColor("#ffffff")
PANEL    = colors.HexColor("#f8fafc")
PANEL2   = colors.HexColor("#eef2ff")
GRID     = colors.HexColor("#cbd5e1")

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm

# ----------------------------------------------------------------------------
# Low-level drawing helpers
# ----------------------------------------------------------------------------
class C:
    def __init__(self, path):
        self.c = canvas.Canvas(path, pagesize=A4)
        self.c.setTitle("SLHA v2 — Schéma, logigramme, plan d'amélioration")
        self.c.setAuthor("Forge CHECKUPAUTO")
        self.page = 1

    def _font(self, bold=False, size=10):
        self.c.setFont(REGB if bold else REG, size)

    def text(self, x, y, s, bold=False, size=10, color=INK, align="left"):
        self._font(bold, size)
        self.c.setFillColor(color)
        if align == "left":
            self.c.drawString(x, y, s)
        elif align == "center":
            self.c.drawCentredString(x, y, s)
        elif align == "right":
            self.c.drawRightString(x, y, s)
        return y

    def wraptext(self, x, y, s, max_w, bold=False, size=10, color=INK, leading=None):
        """Naive word-wrap on pixel width. Returns new y (below last line)."""
        if leading is None:
            leading = size * 1.35
        self._font(bold, size)
        self.c.setFillColor(color)
        words = s.split()
        line = ""
        for w in words:
            trial = (line + " " + w).strip()
            if pdfmetrics.stringWidth(trial, REGB if bold else REG, size) <= max_w:
                line = trial
            else:
                self.c.drawString(x, y, line)
                y -= leading
                line = w
        if line:
            self.c.drawString(x, y, line)
            y -= leading
        return y

    def rect(self, x, y, w, h, fill=None, stroke=INK, sw=0.8, radius=0):
        if fill is not None:
            self.c.setFillColor(fill)
        self.c.setStrokeColor(stroke)
        self.c.setLineWidth(sw)
        if radius:
            self.c.roundRect(x, y, w, h, radius, stroke=1, fill=1 if fill is not None else 0)
        else:
            self.c.rect(x, y, w, h, stroke=1, fill=1 if fill is not None else 0)

    def box(self, x, y, w, h, title, sub=None, fill=PANEL, stroke=INK, tcolor=INK,
            scolor=SUB, tsize=9, ssize=7.5, radius=4, sw=0.9):
        self.rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, radius=radius)
        ty = y + h - tsize - 5
        self.text(x + w / 2, ty, title, bold=True, size=tsize, color=tcolor, align="center")
        if sub:
            self.wraptext_center(x + 4, ty - ssize - 3, w - 8, sub, size=ssize, color=scolor)

    def wraptext_center(self, x, y, max_w, s, bold=False, size=9, color=SUB, leading=None):
        if leading is None:
            leading = size * 1.3
        self._font(bold, size)
        self.c.setFillColor(color)
        words = s.split()
        line = ""
        lines = []
        for w in words:
            trial = (line + " " + w).strip()
            if pdfmetrics.stringWidth(trial, REGB if bold else REG, size) <= max_w:
                line = trial
            else:
                lines.append(line)
                line = w
        if line:
            lines.append(line)
        # draw centered, starting at y going down
        for ln in lines:
            self.c.drawCentredString(x + max_w / 2, y, ln)
            y -= leading
        return y

    def arrow(self, x1, y1, x2, y2, color=INK, sw=1.1, dashed=False, head=5):
        self.c.setStrokeColor(color)
        self.c.setFillColor(color)
        self.c.setLineWidth(sw)
        if dashed:
            self.c.setDash(3, 2)
        self.c.line(x1, y1, x2, y2)
        self.c.setDash()
        # arrowhead
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
        left = (bx + head * 0.55 * math.cos(ang + math.pi / 2),
                by + head * 0.55 * math.sin(ang + math.pi / 2))
        right = (bx + head * 0.55 * math.cos(ang - math.pi / 2),
                 by + head * 0.55 * math.sin(ang - math.pi / 2))
        p = self.c.beginPath()
        p.moveTo(x2, y2)
        p.lineTo(*left)
        p.lineTo(*right)
        p.close()
        self.c.drawPath(p, stroke=0, fill=1)

    def diamond(self, cx, cy, w, h, text, fill=PANEL2, stroke=ACCENT, tsize=8.5):
        self.c.setFillColor(fill)
        self.c.setStrokeColor(stroke)
        self.c.setLineWidth(1.0)
        p = self.c.beginPath()
        p.moveTo(cx, cy + h / 2)
        p.lineTo(cx + w / 2, cy)
        p.lineTo(cx, cy - h / 2)
        p.lineTo(cx - w / 2, cy)
        p.close()
        self.c.drawPath(p, stroke=1, fill=1)
        self.wraptext_center(cx - w / 2 + 6, cy - 2, w - 12, text, bold=True,
                             size=tsize, color=INK, leading=tsize * 1.2)

    def header(self, title, subtitle=None):
        top = PAGE_H - MARGIN
        self.text(MARGIN, top, "SLHA v2", bold=True, size=16, color=ACCENT)
        self.text(MARGIN + 52, top - 1, title, bold=True, size=13, color=INK)
        if subtitle:
            self.text(MARGIN + 52, top - 13, subtitle, size=8.5, color=SUB)
        # rule
        self.c.setStrokeColor(GRID)
        self.c.setLineWidth(0.8)
        self.c.line(MARGIN, top - 18, PAGE_W - MARGIN, top - 18)

    def footer(self):
        self.c.setStrokeColor(GRID)
        self.c.setLineWidth(0.5)
        self.c.line(MARGIN, MARGIN - 4, PAGE_W - MARGIN, MARGIN - 4)
        self._font(False, 7.5)
        self.c.setFillColor(SUB)
        self.c.drawString(MARGIN, MARGIN - 12,
                          "SLHA v2 — Sub-Low Rank Hybrid Attention · Forge CHECKUPAUTO · Édition 2026")
        self.c.drawRightString(PAGE_W - MARGIN, MARGIN - 12, f"p.{self.page}")
        self.page += 1

    def showpage(self):
        self.footer()
        self.c.showPage()

    def save(self):
        self.c.save()


# ===========================================================================
# PAGE 1 — Vue d'ensemble
# ===========================================================================
def page_overview(c: C):
    c.header("Vue d'ensemble du projet",
             "Sub-Low Rank Hybrid Attention — compression de KV-cache pour LLM sur CPU")
    x0 = MARGIN
    y = PAGE_H - MARGIN - 26

    intro = ("SLHA v2 compresse le KV-cache d'un LLM en tuiles de 128 octets calquées sur la "
             "ligne de cache (64 o), pour que l'inférence tienne dans L1/L2/L3 sur CPU plutôt "
             "qu'en VRAM. Chaque clé est décomposée en (1) une base latente de bas rang "
             "quantifiée INT4 et (2) un résidu 1-bit capturé par sign-LSH (Johnson–Lindenstrauss) "
             "et scoré par XOR+popcount. Un gestionnaire élastique (CCOS Soft-Paging) dégrade la "
             "fidélité HOT→WARM→COLD sous pression mémoire sans I/O ni allocation.")
    y = c.wraptext(x0, y, intro, PAGE_W - 2 * MARGIN, size=9.5, color=INK, leading=13.5)
    y -= 6

    # Key facts table
    c.text(x0, y, "Faits clés mesurés (prototype scirust, §7)", bold=True, size=11, color=ACCENT)
    y -= 14
    rows = [
        ("Tuile", "128 o exacts, 0 padding — latent 64 o + résidu 32 o + métadonnées 32 o", "§3.1"),
        ("Score fusionné", "⟨q_coarse, dequant(latent)⟩ + λ·(d_s − 2·popcount(q_sign ⊕ B))", "§2.3"),
        ("Latent", "INT4 signé (zero-point), variantes MX groupé + NF4 — même tuile 128 o", "§5"),
        ("Résidu", "256 bits sign-LSH ; distance de Hamming = produit scalaire ±1", "§2.2"),
        ("SIMD x86", "AVX2 ×11,5 · AVX-512 ×14,1 vs scalaire (Xeon)", "§7.4"),
        ("SIMD ARM", "NEON ×5,7 (Jetson Thor AGX 128, lignes 64 o, sve2 détecté)", "§7.4"),
        ("Bande passante", "~2,5× tokens/s vs clé bf16 (2× moins d'octets/token)", "§7.5"),
        ("Fidélité sortie", "cos(softmax·V) 0,95–0,997 vs attention FP — le softmax absorbe", "§7.6"),
        ("Soft-Paging", "pager ½ tuiles HOT→WARM : sortie à cos 0,9995 (quasi sans perte)", "§4"),
        ("Levier réel #1", "projection bas-rang apprise (SGD task-aware) : WARM 0,16→0,86", "§7.7"),
        ("Faux levier", "largeur de bits : INT8 = INT4 au coarse — le goulot est la projection", "§7.8"),
        ("λ calibrée", "forme ∝σ_E validée ; constante corrigée ×4,2 → C_emp ≈ 0,33", "§7.9"),
    ]
    colw = [70, PAGE_W - 2 * MARGIN - 70 - 28, 24]
    for label, val, ref in rows:
        c.rect(x0, y - 12, colw[0], 13, fill=PANEL2, stroke=GRID, sw=0.5)
        c.rect(x0 + colw[0], y - 12, colw[1], 13, fill=PAPER, stroke=GRID, sw=0.5)
        c.rect(x0 + colw[0] + colw[1], y - 12, colw[2], 13, fill=PANEL, stroke=GRID, sw=0.5)
        c.text(x0 + 4, y - 9, label, bold=True, size=8.5, color=ACCENT)
        c.text(x0 + colw[0] + 4, y - 9, val, size=8.5, color=INK)
        c.text(x0 + colw[0] + colw[1] + 4, y - 9, ref, size=8, color=SUB)
        y -= 13
    y -= 8

    # Components map
    c.text(x0, y, "Architecture du crate scirust (workspace Cargo)", bold=True, size=11, color=ACCENT)
    y -= 14
    mods = [
        ("attention/slha_v2.rs", "Tuile 128 o, codecs INT4/MX/NF4, kernels AVX2/AVX-512/NEON, popcount VPOPCNTDQ", LATENT),
        ("ccos.rs", "ElasticKvCache : arène contiguë, HOT/WARM/COLD, enforce_budget, free-list", HOT),
        ("learned.rs", "PCA (jacobi_eigh) + projection task-aware SGD, whitening, encode/decode", RESID),
        ("metrics.rs · linalg.rs · rng.rs", "cos/spearman/softmax, eigendecomposition, PRNG gaussien", META),
        ("audit.rs + bin/slha_audit.rs", "Auto-audit runtime : layout, équivalence SIMD, fidélité, budget, déterminisme", GREEN),
        ("slha-mcp/", "Serveur MCP stdio zero-dep : 5 outils (audit/explain/compress/score/benchmark)", ACCENT2),
        ("examples/ · scripts/", "measure, calibrate_lambda, ccos_softpaging, platform_report, stress_test", SUB),
    ]
    for name, desc, col in mods:
        c.rect(x0, y - 11, 95, 13, fill=col, stroke=col, radius=2)
        c.text(x0 + 4, y - 8, name, bold=True, size=7.6, color=PAPER)
        c.wraptext(x0 + 100, y - 4, desc, PAGE_W - 2 * MARGIN - 100, size=8.2, color=INK, leading=10)
        y -= 14.5

    c.showpage()


# ===========================================================================
# PAGE 2 — Schéma de principe
# ===========================================================================
def page_principle(c: C):
    c.header("Schéma de principe", "Flux de données : encodage clé, chemin requête, score fusionné, tuile, Soft-Paging")
    W = PAGE_W - 2 * MARGIN

    # ---- Encodage d'une clé (gauche) -------------------------------------
    ytop = PAGE_H - MARGIN - 30
    c.text(MARGIN, ytop, "① Encodage d'une clé K_j (par tête n)", bold=True, size=10, color=ACCENT)
    enc_x = MARGIN
    enc_y = ytop - 14
    bw, bh = 92, 16

    boxes_enc = [
        (enc_x, enc_y,            "Vecteur clé K_j ∈ ℝ^d", "d_model (ex. 256-1024)"),
        (enc_x, enc_y - 30,       "Projection bas-rang W_down", "→ h_KV ∈ ℝ^{d_c=128}"),
        (enc_x, enc_y - 60,       "Quantification INT4 (MX/NF4)", "64 o, 8 échelles de groupe"),
        (enc_x, enc_y - 90,       "Reconstruction K_coarse", "W_up,K^(n) · h_KV"),
        (enc_x, enc_y - 122,      "Résidu  E = K_real − K_coarse", "énergie laissée par la base"),
        (enc_x, enc_y - 152,      "Projection JL aléatoire  Z · E", "Z ∈ ℝ^{256×d}, fixe au démarrage"),
        (enc_x, enc_y - 182,      "Signe  B = sign(Z·E) ∈ {±1}^{256}", "→ bitmap 32 o (4×u64)"),
    ]
    for x, yb, t, s in boxes_enc:
        fill = PANEL2 if "INT4" in t else (colors.HexColor("#f3e8ff") if "sign" in t.lower() or "JL" in t or "Résidu" in t else PANEL)
        c.box(x, yb - bh, bw, bh, t, s, fill=fill, tsize=8.5, ssize=7)
    # arrows down
    for i in range(len(boxes_enc) - 1):
        x1, y1 = boxes_enc[i][0] + bw / 2, boxes_enc[i][1] - bh
        x2, y2 = boxes_enc[i + 1][0] + bw / 2, boxes_enc[i + 1][1] - 2
        c.arrow(x1, y1, x2, y2, color=ACCENT, sw=1.2)

    # ---- Chemin requête (droite) -----------------------------------------
    qx = MARGIN + W - bw
    c.text(qx, ytop, "② Chemin requête Q_i (par tête n)", bold=True, size=10, color=ACCENT2)
    boxes_q = [
        (qx, enc_y,            "Requête Q_i ∈ ℝ^d", "même d_model"),
        (qx, enc_y - 30,       "q_coarse = Q·W_up,K^(n)", "∈ ℝ^{128} (espace latent)"),
        (qx, enc_y - 60,       "q_sign = sign(Q · Zᵀ)", "∈ {±1}^{256} → 4×u64"),
    ]
    for x, yb, t, s in boxes_q:
        c.box(x, yb - bh, bw, bh, t, s, fill=colors.HexColor("#ecfeff"), tcolor=ACCENT2, tsize=8.5, ssize=7)
    for i in range(len(boxes_q) - 1):
        x1, y1 = boxes_q[i][0] + bw / 2, boxes_q[i][1] - bh
        x2, y2 = boxes_q[i + 1][0] + bw / 2, boxes_q[i + 1][1] - 2
        c.arrow(x1, y1, x2, y2, color=ACCENT2, sw=1.2)

    # ---- Score fusionné (centre) -----------------------------------------
    sx = MARGIN + W / 2
    sy = enc_y - 95
    sw, sh = 150, 64
    c.rect(sx - sw / 2, sy - sh / 2, sw, sh, fill=colors.HexColor("#fef9c3"),
           stroke=colors.HexColor("#a16207"), sw=1.2, radius=6)
    c.text(sx, sy + sh / 2 - 12, "③ Score fusionné (eq. 2.3)", bold=True, size=9, color=colors.HexColor("#713f12"), align="center")
    c.wraptext_center(sx - sw / 2 + 6, sy + sh / 2 - 24, sw - 12,
                      "coarse = ⟨q_coarse, dequant(latent)⟩", size=8.3, color=INK)
    c.wraptext_center(sx - sw / 2 + 6, sy + sh / 2 - 35, sw - 12,
                      "+ λ·(d_s − 2·popcount(q_sign ⊕ B))", size=8.3, color=RESID, bold=True)
    c.wraptext_center(sx - sw / 2 + 6, sy + sh / 2 - 48, sw - 12,
                      "WARM → terme binaire droppé (λ=0)", size=7.3, color=SUB)
    # arrows from latent (enc box 3) and bitmap (enc box 7) into score
    c.arrow(boxes_enc[2][0] + bw, boxes_enc[2][1] - bh / 2, sx - sw / 2, sy + 6, color=LATENT, sw=1.0)
    c.arrow(boxes_enc[6][0] + bw, boxes_enc[6][1] - bh / 2, sx - sw / 2, sy - 6, color=RESID, sw=1.0)
    # arrows from query side
    c.arrow(boxes_q[1][0], boxes_q[1][1] - bh / 2, sx + sw / 2, sy + 6, color=ACCENT2, sw=1.0)
    c.arrow(boxes_q[2][0], boxes_q[2][1] - bh / 2, sx + sw / 2, sy - 6, color=ACCENT2, sw=1.0)

    # ---- Tuile mémoire (bas) --------------------------------------------
    ty = enc_y - 215
    c.text(MARGIN, ty, "④ Tuile SLHA v2 — 128 octets = 2 lignes de cache (align 64/128)", bold=True, size=10, color=ACCENT)
    ty -= 12
    tile_y = ty - 40
    tile_h = 34
    segs = [
        (64, "Latent INT4  64 o", "d_c=128 → 64 o (1 ligne pleine)", LATENT, PAPER),
        (32, "Résidu 1-bit  32 o", "256 bits = 4×u64 (AVX-512 / NEON)", RESID, PAPER),
        (32, "Métadonnées  32 o", "scale, λ, σ_E, token, pos, head, flags, group_scales", META, PAPER),
    ]
    tx = MARGIN
    total = sum(s[0] for s in segs)
    scale = (W) / total
    for w, label, sub, col, fill in segs:
        pw = w * scale
        c.rect(tx, tile_y, pw, tile_h, fill=fill, stroke=col, sw=1.2)
        c.rect(tx, tile_y + tile_h, pw, 5, fill=col, stroke=col)
        c.wraptext_center(tx + 3, tile_y + tile_h - 10, pw - 6, label, bold=True, size=8, color=col)
        c.wraptext_center(tx + 3, tile_y + tile_h - 22, pw - 6, sub, size=6.6, color=SUB, leading=8)
        tx += pw
    # cache line markers
    c.c.setStrokeColor(GRID)
    c.c.setLineWidth(0.5)
    c.c.setDash(2, 2)
    half = W / 2
    c.c.line(MARGIN + half, tile_y - 3, MARGIN + half, tile_y + tile_h + 8)
    c.c.setDash()
    c.text(MARGIN, tile_y - 10, "ligne 0 (64 o)", size=6.8, color=SUB)
    c.text(MARGIN + half + 2, tile_y - 10, "ligne 1 (64 o)", size=6.8, color=SUB)

    # ---- Soft-Paging (bas droit / bande) --------------------------------
    py = tile_y - 28
    c.text(MARGIN, py, "⑤ CCOS Soft-Paging — élasticité de fidélité sous budget mémoire",
           bold=True, size=10, color=HOT)
    py -= 16
    state_w, state_h = (W - 30) / 3, 46
    states = [
        ("HOT", "latent 4-bit + résidu 1-bit", "128 o · L1/L2", HOT, colors.HexColor("#fffbeb")),
        ("WARM", "latent seul (résidu droppé, λ=0)", "96 o · L3/DRAM (−25 %)", WARM, colors.HexColor("#fff7ed")),
        ("COLD", "slot recyclé (snapshot EventLog)", "0 o · disque chiffré", COLD, colors.HexColor("#f1f5f9")),
    ]
    sxp = MARGIN
    for i, (t, s, f, col, fill) in enumerate(states):
        c.box(sxp, py - state_h, state_w, state_h, t, f"\n{s}", fill=fill, stroke=col, sw=1.3,
              tsize=11, ssize=7.6, tcolor=col, radius=5)
        if i < 2:
            ax = sxp + state_w
            c.arrow(ax + 1, py - state_h / 2, ax + 9, py - state_h / 2, color=HOT, sw=1.4, head=5)
            c.text(ax + 2, py - state_h / 2 + 7, "page_out" if i == 0 else "evict", size=6.6, color=HOT, bold=True)
        sxp += state_w + 15

    c.showpage()


# ===========================================================================
# PAGE 3 — Logigramme (runtime flow)
# ===========================================================================
def page_flowchart(c: C):
    c.header("Logigramme d'exécution", "Boucle de génération : encodage, gestion budget, scoring, dispatch SIMD")
    W = PAGE_W - 2 * MARGIN

    # Swimlane columns
    col_enc = MARGIN + 4
    col_bud = MARGIN + W * 0.40
    col_scr = MARGIN + W * 0.72
    lane_top = PAGE_H - MARGIN - 28

    # lane headers
    for x, t, col in [(col_enc, "Encodage / insertion", LATENT),
                      (col_bud, "Gestion mémoire (CCOS)", HOT),
                      (col_scr, "Scoring & sortie", ACCENT2)]:
        c.c.setFillColor(col)
        c.c.setStrokeColor(col)
        c.c.setLineWidth(0.6)
        c.rect(x - 4, lane_top - 4, W * 0.27, 14, fill=col, stroke=col, radius=3)
        c.text(x + (W * 0.27) / 2 - 4, lane_top - 1, t, bold=True, size=8.6, color=PAPER, align="center")

    # ---------- ENCODAGE lane ----------
    y = lane_top - 26
    proc_w, proc_h = W * 0.27, 22
    enc_steps = [
        "Nouveau jeton → K_j, V_j (par tête)",
        "h_KV = W_down · K_j   (projection bas-rang)",
        "Quantification INT4/MX/NF4 → 64 o",
        "E = K_j − K_coarse ;  B = sign(Z·E)",
        "σ_E, λ = C·σ_E   (calibration §7.9)",
        "insert() → arène, slot réutilisé si COLD",
    ]
    centers_enc = []
    for s in enc_steps:
        c.rect(col_enc - 4, y - proc_h, proc_w, proc_h, fill=PANEL2, stroke=LATENT, sw=0.9, radius=4)
        c.wraptext_center(col_enc - 2, y - 8, proc_w - 4, s, size=7.8, color=INK, leading=8.6)
        centers_enc.append((col_enc + proc_w / 2 - 4, y))
        y -= proc_h + 8
    for i in range(len(centers_enc) - 1):
        c.arrow(centers_enc[i][0], centers_enc[i][1] - proc_h,
                centers_enc[i + 1][0], centers_enc[i + 1][1], color=LATENT, sw=1.0)

    # arrow from encodage last to scoring lane (top) — long curved via budget lane top
    c.arrow(centers_enc[-1][0] + proc_w / 2, centers_enc[-1][1] - proc_h / 2,
            col_bud - 8, lane_top - 8, color=SUB, sw=0.9, dashed=True)

    # ---------- BUDGET lane ----------
    yb = lane_top - 26
    bud_w = W * 0.27
    # decision: live_bytes > budget ?
    c.diamond(col_bud + bud_w / 2 - 4, yb - 26, 70, 30, "live_bytes\n> budget ?", fill=colors.HexColor("#fef3c7"),
              stroke=HOT, tsize=7.6)
    # Non branch -> straight down (no action)
    c.text(col_bud + bud_w / 2 - 4 + 40, yb - 28, "non", size=7, color=GREEN, bold=True)
    # Oui branch -> page out policy
    yb2 = yb - 70
    c.rect(col_bud - 4, yb2 - proc_h, bud_w, proc_h, fill=PANEL, stroke=HOT, sw=0.9, radius=4)
    c.wraptext_center(col_bud - 2, yb2 - 8, bud_w - 4,
                      "page_out HOT→WARM : plus faible σ_E d'abord (LowestImpactFirst)",
                      size=7.4, color=INK, leading=8.4)
    c.text(col_bud + bud_w / 2 - 4 - 40, yb - 30, "oui", size=7, color=HOT, bold=True)
    c.arrow(col_bud + bud_w / 2 - 4, yb - 41, col_bud + bud_w / 2 - 4, yb2, color=HOT, sw=1.0)

    # second decision: still over budget?
    yb3 = yb2 - 44
    c.diamond(col_bud + bud_w / 2 - 4, yb3, 70, 28, "encore\n> budget ?", fill=colors.HexColor("#fee2e2"),
              stroke=RED, tsize=7.6)
    c.arrow(col_bud + bud_w / 2 - 4, yb2 - proc_h, col_bud + bud_w / 2 - 4, yb3 + 14, color=HOT, sw=1.0)

    # evict
    yb4 = yb3 - 50
    c.rect(col_bud - 4, yb4 - proc_h, bud_w, proc_h, fill=colors.HexColor("#fef2f2"), stroke=RED, sw=0.9, radius=4)
    c.wraptext_center(col_bud - 2, yb4 - 8, bud_w - 4,
                      "evict() → COLD : plus ancien d'abord (causal), slot recyclé",
                      size=7.4, color=INK, leading=8.4)
    c.text(col_bud + bud_w / 2 - 4 - 40, yb3 - 4, "oui", size=7, color=RED, bold=True)
    c.arrow(col_bud + bud_w / 2 - 4, yb3 - 14, col_bud + bud_w / 2 - 4, yb4, color=RED, sw=1.0)
    c.text(col_bud + bud_w / 2 - 4 + 40, yb3 - 4, "non", size=7, color=GREEN, bold=True)

    # O(1) masking note
    c.wraptext(col_bud - 4, yb4 - proc_h - 8,
               "page_out = O(1) : zero 32 o + drapeau, sans I/O ni allocation",
               bud_w + 8, size=7, color=SUB, leading=8.5)

    # ---------- SCORING lane ----------
    ys = lane_top - 26
    scr_w = W * 0.27
    scr_steps = [
        "Requête Q_i → q_coarse, q_sign",
        "Pour chaque tuile live (non COLD) :",
    ]
    centers_scr = []
    for s in scr_steps:
        c.rect(col_scr - 4, ys - proc_h, scr_w, proc_h, fill=colors.HexColor("#ecfeff"), stroke=ACCENT2, sw=0.9, radius=4)
        c.wraptext_center(col_scr - 2, ys - 8, scr_w - 4, s, size=7.8, color=INK, leading=8.6)
        centers_scr.append((col_scr + scr_w / 2 - 4, ys))
        ys -= proc_h + 8

    # decision: tile WARM ?
    c.diamond(col_scr + scr_w / 2 - 4, ys - 22, 64, 28, "tile WARM ?", fill=PANEL, stroke=ACCENT2, tsize=7.8)
    ys -= 50
    # yes -> coarse only
    c.rect(col_scr - 4, ys - proc_h, scr_w, proc_h, fill=colors.HexColor("#fff7ed"), stroke=WARM, sw=0.9, radius=4)
    c.wraptext_center(col_scr - 2, ys - 8, scr_w - 4, "score = terme coarse seul (résidu droppé)", size=7.4, color=INK, leading=8.4)
    c.text(col_scr + scr_w / 2 - 4 - 36, ys + 14, "oui", size=7, color=WARM, bold=True)
    c.arrow(col_scr + scr_w / 2 - 4, ys + 22, col_scr + scr_w / 2 - 4, ys, color=WARM, sw=1.0)

    # no -> dispatch SIMD
    ys -= proc_h + 10
    c.text(col_scr + scr_w / 2 - 4 + 36, ys + proc_h + 18, "non", size=7, color=ACCENT2, bold=True)
    c.rect(col_scr - 4, ys - proc_h, scr_w, proc_h, fill=PANEL2, stroke=ACCENT2, sw=0.9, radius=4)
    c.wraptext_center(col_scr - 2, ys - 8, scr_w - 4, "compute_score (HOT) — coarse + λ·résidu", size=7.4, color=INK, leading=8.4)
    ys -= proc_h + 8

    # dispatch box
    disp_h = 34
    c.rect(col_scr - 4, ys - disp_h, scr_w, disp_h, fill=colors.HexColor("#eef2ff"), stroke=ACCENT, sw=1.0, radius=4)
    c.wraptext_center(col_scr - 2, ys - 10, scr_w - 4, "Dispatch SIMD runtime", bold=True, size=7.6, color=ACCENT, leading=8)
    c.wraptext_center(col_scr - 2, ys - 20, scr_w - 4,
                      "AVX-512 VPOPCNTDQ > AVX2 > scalaire  (x86)", size=6.8, color=INK, leading=7.6)
    c.wraptext_center(col_scr - 2, ys - 28, scr_w - 4,
                      "NEON + CNT  (AArch64, sve2 quand stable)", size=6.8, color=INK, leading=7.6)
    ys -= disp_h + 8

    # softmax -> output
    c.rect(col_scr - 4, ys - proc_h, scr_w, proc_h, fill=colors.HexColor("#dcfce7"), stroke=GREEN, sw=0.9, radius=4)
    c.wraptext_center(col_scr - 2, ys - 8, scr_w - 4,
                      "softmax(scores/√d) · V  →  sortie (cos 0,95–0,997)", size=7.2, color=INK, leading=8.2)

    # cross-lane arrow: budget done -> scoring
    c.arrow(col_bud + bud_w - 4, lane_top - 16, col_scr - 6, lane_top - 16, color=SUB, sw=1.0, dashed=True)

    # audit/mcp side note (bottom)
    ny = MARGIN + 26
    c.rect(MARGIN, ny, W, 22, fill=colors.HexColor("#f0fdf4"), stroke=GREEN, sw=0.8, radius=4)
    c.wraptext(MARGIN + 6, ny + 15, "Audit & agent :  slha-audit (layout, équivalence SIMD, fidélité, budget, déterminisme → rapport JSON/MD)  ·  "
               "slha-mcp expose audit/explain/compress/score/benchmark aux LLM-agents.",
               W - 12, size=7.6, color=INK, leading=9)

    c.showpage()


# ===========================================================================
# PAGES 4+ — Plan d'amélioration
# ===========================================================================
# Axes: (id, titre, probleme, levier, papers, priorite, effort, gain_attendu)
# Citations vérifiées (arXiv, 2020-2026) issues de la revue de littérature périphérique.
AXES = [
    ("A1", "Projection bas-rang sur clés PRE-RoPE",
     "Le plafond de fidélité du coarse tient à la projection (§7.8). Or RoPE *mélange* les canaux et "
     "crée des outliers : une projection post-RoPE est structurellement sous-optimale.",
     "Appliquer W_down sur les clés PRE-RoPE (avant rotation), reconstruire puis post-rotater. "
     "ShadowKV montre que les clés pre-RoPE sont *bien* plus low-rank que post-RoPE : c'est "
     "probablement le plus gros gain de fidélité latent disponible, et il explique le goulot mesuré.",
     "ShadowKV (arXiv 2410.21465), KVQuant pre-RoPE (2401.18079).",
     "Critique", "Moyen", "Lève le plafond WARM à la source — explique le goulot « projection » du §7.8."),

    ("A2", "Blanchiment par incohérence Hadamard (QuIP# / Palu)",
     "Le résidu 1-bit perd de la résolution sur directions quasi-orthogonales (Spearman 0,67, §7.1) ; "
     "l'énergie résiduelle est concentrée sur quelques canaux (outliers).",
     "Transformée de Hadamard aléatoire (RHT = Hadamard × diagonale ±1) appliquée à E *et* Q avant "
     "le sign-LSH : « incoherence processing » qui uniformise l'énergie, chaque bit capte autant "
     "d'info, outliers supprimés. FWHT en O(n log n), entrées ±1 (pas de FP mul). Appliquer aussi au latent.",
     "QuIP# (2402.04396), Palu Walsh-Hadamard fused (2407.21118), NSNQuant (2505.18231).",
     "Critique", "Moyen", "Meilleure résolution du résidu + latent ; tombe pile sur le goulot mesuré."),

    ("A3", "Projection entraînée (Cayley orthogonale) + Matryoshka",
     "La SGD task-aware (§7.7) n'est validée que sur données synthétiques, P seule, sans vrai modèle. "
     "MatryoshkaKV confirme que PCA est sous-optimal pour les LLM (non-linéarité).",
     "Fine-tuner W_down/W_up conjointement avec un vrai LLM, paramétrisation Cayley (orthogonale : "
     "produits scalaires préservés par construction), init PCA. Entraînement Matryoshka (rangs "
     "aléatoires) : un seul modèle sert toute l'échelle HOT/WARM. Fusionner W_up dans la projection "
     "de sortie (Palu) pour une reconstruction zéro-coût runtime.",
     "MatryoshkaKV (2410.14731), Palu (2407.21118), StiefAttention (2601.21686), Linformer (2006.04768).",
     "Critique", "Élevé", "Lève le plafond WARM (0,16→0,86 déjà montré) sur vrai modèle ; multi-rang natif."),

    ("A4", "Résidu multi-bit + LSH multi-round (Soft-Paging gradué)",
     "256 bits sign-LSH = 1 bit/hyperplan : modeste. Le résidu est paginé binairement (WARM on/off), "
     "pas gradué. Reformer montre qu'un seul bitmap a un fort taux de faux-négatifs.",
     "(a) Multi-round LSH : 2-3 bitmaps indépendants, vote majoritaire ; (b) résidu additif/RVQ : un "
     "niveau ±1 + un niveau fin sur le sous-espace à forte σ_E. Soft-Paging devient gradué "
     "(HOT2→HOT1→WARM). Codebook partagé pré-calculé (NSNQuant) pour économiser la mémoire résiduelle. "
     "Normalisation shared-QK avant signing (cohérence hash query↔key).",
     "Reformer LSH (2001.04451), RBE/BEBR (1802.06466, 2302.08714), QINCo (ICML 2024), NSNQuant (2505.18231).",
     "Haute", "Moyen", "Fidélité graduée + meilleur HOT à rho élevé ; Soft-Paging plus fin."),

    ("A5", "Éviction informée (heavy-hitters / sinks / observation window)",
     "L'éviction COLD est purement causale (plus ancien d'abord). Or l'attention a des heavy-hitters "
     "et des sinks (premiers tokens) qu'évincer détruit la qualité. σ_E seul ignore l'importance réelle.",
     "Politique d'importance : score cumulé par token (H2O), préservation des attention sinks "
     "(StreamingLLM — pinner les premiers tokens en HOT), fenêtre d'observation pré-décodage (SnapKV) "
     "pour un budget par tête, pyramide par couche (PyramidKV : +de KV en couches basses). σ_E devient "
     "un signal complémentaire, pas seul critère. Profiling par tête une fois (FastGen) ⇒ λ et budget par tête.",
     "H2O (2306.14048), StreamingLLM (2309.17453), SnapKV (2404.14469), PyramidKV (2406.02069), FastGen (2310.01801).",
     "Haute", "Faible", "Moins de dérive sous forte pression ; cohérent avec la pratique LLM 2024-26."),

    ("A6", "Quantification asymétrique K/V (per-channel K, per-token V)",
     "NF4/MX réduisent l'erreur de reconstruction mais le gain end-to-end est marginal (§7.8). KIVI "
     "montre que K et V ont des statistiques *différentes* : K a des outliers par canal, V non.",
     "Traitement asymétrique : latent h_K quantifié par canal, h_V par token. Fenêtre résiduelle FP "
     "glissante (KIVI) comme baseline à comparer au bitmap 1-bit. Extraction des canaux outliers en "
     "FP/sparse (Atom, GEAR) si leur nombre tient dans le budget tuile. Réordonnancement de canaux "
     "pour isoler les outliers (Atom) — compatible avec le tuilage cache.",
     "KIVI (2402.02750), Atom (2310.19102), GEAR (2403.05527), QAQ (2403.04643), AWQ.",
     "Haute", "Moyen", "Réduit l'erreur latent sur vrais modèles ; GEAR = analogie externe du système complet."),

    ("A7", "Validation matérielle réelle (perf + perplexité)",
     "Compteurs de cache (§6.1) et perplexité d'un vrai modèle (§6.3) restent non mesurés (sandbox). "
     "Les ratios SIMD sont indicatifs (banc partagé).",
     "Runner perf stat (L1/L2/LLC misses) sous Debian 13 ; intégrer à un vrai décodeur (greffon "
     "llama.cpp/vLLM) pour TTFT, débit bout-en-bout et Δperplexité HOT/WARM/COLD. slha-audit peut "
     "collecter ces compteurs en CI sur metal. Sans cela, les cibles §6 restent des hypothèses.",
     "Bancs vLLM, perf_event, KV-Runahead.",
     "Haute", "Moyen", "Transforme les hypothèses §6 en résultats ; crédibilité du papier."),

    ("A8", "Pagination KV par blocs (PagedAttention) + prefetch scheduler",
     "L'arène contiguë de ccos.rs recycle des slots mais ne pagine pas par blocs fixes ; pas de "
     "partage de KV entre requêtes (prefix sharing) ni de fragmentation contrôlée.",
     "Tuile 128 o = page naturelle. Tables de pages + mapping logique→physique (relocation HOT→WARM "
     "sans copier le latent), copy-on-write pour prefix sharing multi-requête, prefetch "
     "scheduler-aware (réchauffer les bitmaps droppés depuis la file de jobs). CCOS devient un paged "
     "KV manager au-dessus des tuiles.",
     "vLLM/PagedAttention (2309.06180), AttentionStore (2403.19708), ChunkAttention (2402.15220).",
     "Moyenne", "Moyen", "Multi-requête, partage de préfixe, fragmentation contrôlée — usage serveur."),

    ("A9", "Top-k approx + re-rank two-pass",
     "On score *toutes* les tuiles live. À très long contexte, même 128 o/token coûte.",
     "Passe 1 coarse-only (résidu droppé) pour filtrer un top-k candidat, passe 2 HOT (résidu) pour "
     "reranker. Soft-Paging = filtre naturel : WARM = passe 1, HOT = passe 2. Réduit le travail de "
     "popcount au top-k. ChunkAttention donne le pattern de merge softmax en ligne pour le batching.",
     "Reformer LSH filtering (2001.04451), ScaNN anisotropic + re-rank, ChunkAttention (2402.15220).",
     "Moyenne", "Moyen", "Débit long-contexte ; cohérent avec la sémantique HOT/WARM."),

    ("A10", "LSH appris / anisotrope + normalisation shared-QK",
     "Z est une projection JL Gaussienne *fixe* — sous-optimale pour la distribution réelle des "
     "résidus. Reformer normalise Q/K partagés avant le hash pour la cohérence.",
     "Apprendre Z (ou un codebook binaire) pour maximiser corrélation sign(Z·E) ↔ cos(E, requête), "
     "SGD ou PCA signée, conjoint avec A3. LSH anisotrope (préférence aux voisins de forte similarité, "
     "pas cosinus uniforme). Normaliser Q/K avant signing (cohérence query↔key bitmap).",
     "ScaNN (anisotropic quantization), Reformer shared-QK (2001.04451), BOLT, NSNQuant (2505.18231).",
     "Moyenne", "Moyen", "Meilleur Spearman du cœur binaire (0,67 → ?) sans plus de bits."),

    ("A11", "Chemin SVE2 / scalable vector pour ARM",
     "SVE2 est présent sur la cible Jetson Thor mais non exploité : intrinsèques nightly-only en Rust, "
     "chemin NEON+cnt livré seulement.",
     "Ajouter un chemin SVE2 via asm! manuel (vectorisation scalable) validé *sur appareil* (test "
     "d'équivalence bit-à-bit comme les autres chemins), repli NEON. Surveiller la stabilisation des "
     "intrinsèques core::arch::aarch64.",
     "Spécification ARM SVE2 ; std::simd Rust.",
     "Moyenne", "Élevé", "Gain ARM supplémentaire (largeur de vecteur adaptive) ; route serveur Neoverse."),

    ("A12", "Kernel NF4 LUT + format MX (finition)",
     "La déquant NF4 coûte 21-40 % de la latence (LUT 16 niveaux). Les MX group scales peuvent "
     "déborder. Gain end-to-end NF4 marginal (§7.8) : à ne prioriser qu'après A1-A3.",
     "LUT NF4 en mémoire partagée (Fast NF4 kernel, ~2×). Group scales E8M0 (OCP MX) ou FP8 partagé "
     "asymétrique (AMXFP4) ; Overflow-Aware Scaling (OAS) pour le débordement logiciel. Décision "
     "pilotée par la mesure, comme les codecs existants.",
     "Fast NF4 dequant (2604.02556), AMXFP4 (2411.09909), OAS+MBS (2603.08713), OCP MX (ARITH 2025).",
     "Basse", "Faible", "Accélère le chemin NF4 + robustesse des group scales ; gain fidélité faible."),
]

def page_plan_intro(c: C):
    c.header("Plan d'amélioration", "Axes priorisés · grounding littérature · roadmap")
    W = PAGE_W - 2 * MARGIN
    y = PAGE_H - MARGIN - 26

    intro = ("Le projet a validé la *mécanique* (suite de tests automatisés, équivalences SIMD, Soft-Paging quasi sans "
             "perte, sortie d'attention cos 0,95–0,997) et identifié, par la mesure, ses *vrais* "
             "leviers : la projection bas-rang (A1) et le résidu 1-bit (A2/A3) — la quantification du "
             "latent n'est pas le goulot. Le plan ci-dessous priorise donc la fidélité et la politique "
             "de cache, avant l'optimisation matérielle, et ancre chaque axe dans la littérature "
             "périphérique 2022-2026.")
    y = c.wraptext(MARGIN, y, intro, W, size=9.5, color=INK, leading=13)
    y -= 6

    # priority table header
    c.text(MARGIN, y, "Tableau des axes (priorité / effort)", bold=True, size=11, color=ACCENT)
    y -= 6
    cols = [(28, "ID"), (150, "Axe"), (52, "Priorité"), (42, "Effort"), (W - 28 - 150 - 52 - 42, "Gain attendu")]
    hy = y - 12
    cx = MARGIN
    for w, t in cols:
        c.rect(cx, hy, w, 14, fill=ACCENT, stroke=ACCENT)
        c.text(cx + 3, hy + 4, t, bold=True, size=8.2, color=PAPER)
        cx += w
    y = hy - 2
    pcolor = {"Critique": RED, "Haute": HOT, "Moyenne": ACCENT2, "Basse": SUB}
    for aid, title, _prob, _lev, _papers, prio, eff, gain in AXES:
        cx = MARGIN
        rh = 16
        c.rect(cx, y - rh, cols[0][0], rh, fill=PANEL2, stroke=GRID, sw=0.5)
        c.text(cx + 3, y - rh + 5, aid, bold=True, size=8.2, color=ACCENT)
        cx += cols[0][0]
        c.rect(cx, y - rh, cols[1][0], rh, fill=PAPER, stroke=GRID, sw=0.5)
        c.wraptext(cx + 3, y - rh + 12, title, cols[1][0] - 6, bold=True, size=7.8, color=INK, leading=8.6)
        cx += cols[1][0]
        c.rect(cx, y - rh, cols[2][0], rh, fill=PAPER, stroke=GRID, sw=0.5)
        c.text(cx + 3, y - rh + 5, prio, bold=True, size=7.8, color=pcolor.get(prio, INK))
        cx += cols[2][0]
        c.rect(cx, y - rh, cols[3][0], rh, fill=PAPER, stroke=GRID, sw=0.5)
        c.text(cx + 3, y - rh + 5, eff, size=7.8, color=INK)
        cx += cols[3][0]
        c.rect(cx, y - rh, cols[4][0], rh, fill=PAPER, stroke=GRID, sw=0.5)
        c.wraptext(cx + 3, y - rh + 12, gain, cols[4][0] - 6, size=7.4, color=SUB, leading=8.4)
        y -= rh

    y -= 8
    c.text(MARGIN, y, "Lecture : Critique = lève le plafond de fidélité ; Haute = gain mesurable direct ; "
                       "Moyenne/Basse = optimisation ou validation.", size=8, color=SUB)
    c.showpage()


def page_plan_detail(c: C, start_idx, page_no):
    c.header(f"Plan d'amélioration — détail ({page_no})", "Problème · Levier · Grounding littérature")
    W = PAGE_W - 2 * MARGIN
    y = PAGE_H - MARGIN - 24
    chunk = AXES[start_idx:start_idx + 2]
    for aid, title, prob, lev, papers, prio, eff, gain in chunk:
        # header bar
        c.rect(MARGIN, y - 16, W, 16, fill=ACCENT, stroke=ACCENT, radius=2)
        c.text(MARGIN + 5, y - 12, f"{aid} — {title}", bold=True, size=9.5, color=PAPER)
        c.text(MARGIN + W - 60, y - 12, f"{prio} · effort {eff}", bold=True, size=7.6,
               color=colors.HexColor("#fef9c3"))
        y -= 22
        # problème
        c.text(MARGIN, y, "Problème", bold=True, size=8.2, color=RED)
        y = c.wraptext(MARGIN + 40, y + 1, prob, W - 40, size=8.3, color=INK, leading=10.5) - 2
        # levier
        c.text(MARGIN, y, "Levier", bold=True, size=8.2, color=GREEN)
        y = c.wraptext(MARGIN + 40, y + 1, lev, W - 40, size=8.3, color=INK, leading=10.5) - 2
        # papers
        c.text(MARGIN, y, "Littérature", bold=True, size=8.2, color=ACCENT2)
        y = c.wraptext(MARGIN + 40, y + 1, papers, W - 40, size=8.3, color=ACCENT2, leading=10.5) - 2
        # gain
        c.text(MARGIN, y, "Gain attendu", bold=True, size=8.2, color=HOT)
        y = c.wraptext(MARGIN + 40, y + 1, gain, W - 40, size=8.3, color=INK, leading=10.5) - 2
        y -= 10
        if y < MARGIN + 60:
            c.showpage()
            return True, start_idx + 2
    c.showpage()
    return True, start_idx + 2


def page_roadmap(c: C):
    c.header("Roadmap & honnêteté", "Phases · dépendances · limites assumées")
    W = PAGE_H - 2 * MARGIN  # not used; keep W as page width
    W = PAGE_W - 2 * MARGIN
    y = PAGE_H - MARGIN - 26

    c.text(MARGIN, y, "Phases proposées (12-18 mois)", bold=True, size=11, color=ACCENT)
    y -= 16
    phases = [
        ("Phase 1 — Fidélité (A1, A2, A3)",
         "Entraînement conjoint de la projection + incohérence Hadamard + résidu multi-bit. "
         "Objectif : lever le plafond WARM et graduer le Soft-Paging. Livrable : projection entraînée "
         "sur un petit LLM, banc de fidélité score/sortie/perplexité."),
        ("Phase 2 — Politique de cache (A4, A5, A9)",
         "Éviction informée (heavy-hitters/sinks) + pagination par blocs (PagedAttention) + two-pass "
         "top-k. Objectif : long contexte et multi-requête serveur avec moins de dérive. "
         "Livrable : ElasticKvCache v2 sur tuiles-pages."),
        ("Phase 3 — Validation matérielle (A8, A6)",
         "Intégration greffon llama.cpp/vLLM, perf stat (cache misses), perplexité réelle, chemin SVE2 "
         "validé sur appareil. Objectif : transformer les hypothèses §6 en résultats mesurés."),
        ("Phase 4 — Finition (A7, A10)",
         "Quantification asymétrique/outliers et LSH appris — uniquement après que A1-A3 ont montré "
         "leur plafond. Décisions pilotées par la mesure."),
    ]
    for t, d in phases:
        c.rect(MARGIN, y - 12, W, 12, fill=PANEL2, stroke=GRID, sw=0.5)
        c.text(MARGIN + 4, y - 9, t, bold=True, size=8.6, color=ACCENT)
        y -= 16
        y = c.wraptext(MARGIN + 8, y, d, W - 8, size=8.4, color=INK, leading=10.8) - 6

    # honnêteté / limites
    y -= 4
    c.text(MARGIN, y, "Limites assumées & principes à conserver", bold=True, size=11, color=RED)
    y -= 16
    limits = [
        "Les valeurs §6 (≥85 % cache misses, 3,5-5×, ΔP<0,04) restent des hypothèses tant que A8 n'est pas fait.",
        "Les ratios SIMD sont indicatifs (banc partagé) — mesurer les vôtres ; ne pas les présenter comme résultats.",
        "Ne pas réintroduire un faux levier (largeur de bits) : la projection est le goulot, mesuré (§7.8).",
        "Tuile 128 o et invariant zéro-padding sont l'actif matériel — toute évolution doit préserver l'alignement cache.",
        "Toute technique empruntée (Hadamard, RVQ, heavy-hitters) doit être validée par test d'équivalence + banc, "
        "comme les chemins SIMD existants — pas ajoutée sur foi de la littérature.",
    ]
    for s in limits:
        y = c.wraptext(MARGIN + 6, y, "• " + s, W - 6, size=8.4, color=INK, leading=10.8) - 3

    c.showpage()


# ===========================================================================
# Main
# ===========================================================================
def main():
    out = os.path.join(os.path.dirname(__file__), "SLHAv2_schema_plan.pdf")
    c = C(out)
    page_overview(c)
    page_principle(c)
    page_flowchart(c)
    page_plan_intro(c)
    # detail pages, 2 axes each
    idx = 0
    pno = 1
    while idx < len(AXES):
        _, idx = page_plan_detail(c, idx, pno)
        pno += 1
    page_roadmap(c)
    c.save()
    print(f"wrote {out}  ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()