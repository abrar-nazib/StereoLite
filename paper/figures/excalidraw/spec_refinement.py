"""One recurrent tile-refinement iteration (RAFT update-block idiom, terse labels)."""
OUT = "refinement"

BLUE_F, BLUE_S = "#dbeafe", "#1e40af"
YEL_F, YEL_S = "#fef3c7", "#b45309"
ORG_F, ORG_S = "#fed7aa", "#c2410c"
GREY_F, GREY_S = "#e2e8f0", "#475569"
BOX_F, BOX_S = "#f8fafc", "#64748b"
INK, SUB = "#1e293b", "#475569"

E = []


def rect(i, x, y, w, h, fill, stroke, text=None, size=12, sw=1.3, round=True, tcolor=None, dash=False):
    d = dict(kind="rect", id=i, x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw, round=round, dash=dash)
    if text:
        d.update(text=text, size=size, tcolor=tcolor or INK)
    E.append(d)


def txt(i, x, y, text, size=12, color=INK, align="left", w=None):
    d = dict(kind="text", id=i, x=x, y=y, text=text, size=size, color=color, align=align)
    if w:
        d["w"] = w
    E.append(d)


def bars(prefix, x, yc, heights, w=12, gap=9, fill=BLUE_F, stroke=BLUE_S):
    for k, h in enumerate(heights):
        rect(f"{prefix}{k}", x + k * (w + gap), yc - h / 2, w, h, fill, stroke, round=False, sw=1.2)


def arrow(i, pts, x=0, y=0, color=INK, sw=1.4, dash=False, head="arrow"):
    E.append(dict(kind="arrow", id=i, x=x, y=y, points=pts, color=color, sw=sw, dash=dash, head=head))


def circ(i, x, y, r, text, size=12, fill="#ffffff", stroke=INK):
    E.append(dict(kind="ellipse", id=i, x=x - r, y=y - r, w=2 * r, h=2 * r, fill=fill, stroke=stroke, sw=1.3,
                  text=text, size=size, tcolor=INK))


def jdot(i, x, y, r=3.5):
    E.append(dict(kind="ellipse", id=i, x=x - r, y=y - r, w=2 * r, h=2 * r, fill=INK, stroke=INK, sw=1))


# ---------------------------------------------------------------- inputs (left column)
bars("fL", 60, 110, [96])
txt("fL_l", 40, 166, "fL", size=11, align="center", w=52)
bars("fR", 60, 250, [96])
txt("fR_l", 40, 306, "fR", size=11, align="center", w=52)
rect("T", 30, 352, 120, 42, BOX_F, BOX_S, text="T = (d, sx, sy, c)", size=11)
txt("T_l", 30, 400, "tile state", size=10, color=SUB, align="center", w=120)

# ---------------------------------------------------------------- warp + local correlation
rect("warp", 150, 272, 76, 48, GREY_F, GREY_S, text="Warp", size=12)
arrow("a_fR_warp", [[0, 0], [38, 0], [38, 46], [76, 46]], x=72, y=250)
arrow("a_d_warp", [[0, 0], [0, -14], [48, -14], [48, -30]], x=140, y=352, dash=True, color=SUB, sw=1.1)   # d from the state
txt("a_d_l", 194, 328, "d", size=11)

rect("corr", 270, 186, 104, 58, GREY_F, GREY_S, text="Local Corr\n±2 px", size=11)
# f_L: right along y=110, junction, one branch down into the corr left side, one on to the concat
arrow("a_fL_line", [[0, 0], [178, 0]], x=72, y=110, head=None)
jdot("j_fL", 250, 110)
arrow("a_fL_corr", [[0, 0], [0, 90], [18, 90]], x=250, y=110)
arrow("a_fL_cat", [[0, 0], [180, 0], [180, 89]], x=250, y=110)
# warped f_R enters the corr from below
arrow("a_warp_corr", [[0, 0], [94, 0], [94, -50]], x=228, y=296)

# ---------------------------------------------------------------- concatenation node
circ("cat", 430, 215, 14, "C", size=12)
arrow("a_corr_cat", [[0, 0], [40, 0]], x=376, y=215)
arrow("a_T_cat", [[0, 0], [280, 0], [280, -144]], x=150, y=373, color=SUB, sw=1.1)            # state into concat from below
arrow("a_ctx_cat", [[0, 0], [0, -84]], x=455, y=360, dash=True, color=YEL_S, sw=1.2)          # context (dashed, yellow)
txt("ctx_l", 440, 362, "ctx", size=11, color=YEL_S)

# ---------------------------------------------------------------- ConvGRU
rect("gru", 486, 160, 110, 110, BOX_F, ORG_S, sw=1.3)
bars("gru_b", 515, 206, [56, 38, 56], w=12, gap=10, fill=ORG_F, stroke=ORG_S)
txt("gru_t", 486, 246, "ConvGRU", size=11, align="center", w=110)
arrow("a_cat_gru", [[0, 0], [40, 0]], x=446, y=215)
arrow("a_h", [[0, 0], [0, -30], [-70, -30], [-70, 0]], x=576, y=160, color=ORG_S, sw=1.3)
txt("h_l", 534, 112, "h", size=11, color=ORG_S)

# ---------------------------------------------------------------- head, gate, update
rect("head", 636, 190, 60, 50, GREY_F, GREY_S, text="Head", size=11)
arrow("a_gru_head", [[0, 0], [38, 0]], x=598, y=215)
txt("dlt_l", 698, 178, "Δd, Δs, Δc", size=10, color=SUB, align="center", w=84)
circ("gate", 740, 215, 14, "⊗", size=13)
arrow("a_head_gate", [[0, 0], [28, 0]], x=698, y=215)
# gate weights: sigma of the hidden state
circ("sig", 740, 300, 14, "σ", size=13)
arrow("a_gru_sig", [[0, 0], [0, 30], [185, 30]], x=541, y=270, color=SUB, sw=1.1)
arrow("a_sig_gate", [[0, 0], [0, -56]], x=740, y=286, color=SUB, sw=1.1)
circ("add", 800, 215, 14, "⊕", size=14)
arrow("a_gate_add", [[0, 0], [30, 0]], x=756, y=215)
arrow("a_T_add", [[0, 0], [650, 0], [650, -145]], x=150, y=388, color=SUB, sw=1.1)           # previous state into the adder
arrow("a_out", [[0, 0], [48, 0]], x=816, y=215)
txt("out_l", 868, 206, "T′", size=13)

ELEMENTS = E
