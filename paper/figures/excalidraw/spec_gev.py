"""Narrow-band geometry encoding volume and fail-soft fusion (IGEV idiom, terse labels)."""
OUT = "gev_fusion"

BLUE_F, BLUE_S = "#dbeafe", "#1e40af"
YEL_F, YEL_S = "#fef3c7", "#b45309"
PINK_F, PINK_S = "#fecaca", "#b91c1c"
GCUBE = ("#a7f3d0", "#6ee7b7", "#34d399", "#047857")
GREY_F, GREY_S = "#e2e8f0", "#475569"
BOX_F, BOX_S = "#f8fafc", "#64748b"
INK, SUB, RED = "#1e293b", "#475569", "#dc2626"

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


def cube(prefix, x, y, w, h, dep, cols):
    fr, tp, sd, st = cols
    E.append(dict(kind="poly", id=prefix + "_top", x=x, y=y - dep, fill=tp, stroke=st, sw=1.2,
                  points=[[0, dep], [dep, 0], [w + dep, 0], [w, dep]]))
    E.append(dict(kind="poly", id=prefix + "_side", x=x + w, y=y - dep, fill=sd, stroke=st, sw=1.2,
                  points=[[0, dep], [dep, 0], [dep, h], [0, h + dep]]))
    E.append(dict(kind="rect", id=prefix + "_front", x=x, y=y, w=w, h=h, fill=fr, stroke=st, sw=1.2, round=False))


def arrow(i, pts, x=0, y=0, color=INK, sw=1.4, dash=False, head="arrow"):
    E.append(dict(kind="arrow", id=i, x=x, y=y, points=pts, color=color, sw=sw, dash=dash, head=head))


def circ(i, x, y, r, text, size=12, fill="#ffffff", stroke=INK):
    E.append(dict(kind="ellipse", id=i, x=x - r, y=y - r, w=2 * r, h=2 * r, fill=fill, stroke=stroke, sw=1.3,
                  text=text, size=size, tcolor=INK))


def dot(i, x, y, r=6, fill=RED):
    E.append(dict(kind="ellipse", id=i, x=x - r, y=y - r, w=2 * r, h=2 * r, fill=fill, stroke=fill, sw=1))


def jdot(i, x, y, r=3.5):
    E.append(dict(kind="ellipse", id=i, x=x - r, y=y - r, w=2 * r, h=2 * r, fill=INK, stroke=INK, sw=1))


Y = 204   # main flow line

# ---------------------------------------------------------------- inputs: 1/4 features
bars("fL", 50, 140, [90])
txt("fL_l", 30, 192, "fL", size=11, align="center", w=52)
bars("fR", 50, 270, [90])
txt("fR_l", 30, 322, "fR", size=11, align="center", w=52)
txt("q_l", 30, 84, "1/4", size=10, color=SUB, align="center", w=52)
arrow("a_fL_cube", [[0, 0], [60, 0], [60, 44], [166, 44]], x=62, y=140)
arrow("a_fR_cube", [[0, 0], [60, 0], [60, -44], [166, -44]], x=62, y=270)

# ---------------------------------------------------------------- narrow-band GEV cube
cube("gev", 230, 162, 96, 84, 24, GCUBE)
txt("gev_l", 190, 112, "Narrow-Band GEV", size=11, align="center", w=180)

# the band: disparity axis below the cube, window of 33 hypotheses centred on the tile disparity d
AX = 278
E.append(dict(kind="rect", id="band_full", x=AX - 8, y=290, w=16, h=120, fill="#f1f5f9", stroke="#cbd5e1", sw=1, round=False))
E.append(dict(kind="rect", id="band_win", x=AX - 8, y=322, w=16, h=56, fill=GCUBE[1], stroke=GCUBE[3], sw=1.2, round=False))
E.append(dict(kind="line", id="band_tick", x=AX - 14, y=350, points=[[0, 0], [28, 0]], color=INK, sw=1.4))
txt("band_d", AX + 18, 342, "d", size=12)
txt("band_p", AX + 18, 314, "+16", size=9.5, color=SUB)
txt("band_m", AX + 18, 370, "−16", size=9.5, color=SUB)
arrow("a_band_cube", [[0, 0], [0, -40]], x=AX, y=290)
arrow("a_d_band", [[0, 0], [0, -40]], x=AX, y=452, dash=True, color=SUB, sw=1.1)
txt("a_d_band_l", AX - 60, 456, "d  (tile state)", size=10, color=SUB, align="center", w=120)

# ---------------------------------------------------------------- regularization and read-out
arrow("a_cube_3d", [[0, 0], [40, 0]], x=352, y=Y)
bars("reg", 398, Y, [60, 60, 60], w=11, gap=8, fill=PINK_F, stroke=PINK_S)
txt("reg_l", 372, 242, "3D Conv", size=10, color=SUB, align="center", w=100)
arrow("a_3d_soft", [[0, 0], [40, 0]], x=456, y=Y)
rect("soft", 500, Y - 22, 88, 44, GREY_F, GREY_S, text="softmax", size=11)
dot("dgev", 612, Y)
# fan-out: three read-outs, labels above each arrow
arrow("a_ro_c", [[0, 0], [120, 0]], x=588, y=Y)
arrow("a_ro_d", [[0, 0], [34, 0], [34, -46], [120, -46]], x=588, y=Y, sw=1.2)
arrow("a_ro_g", [[0, 0], [34, 0], [34, 46], [120, 46]], x=588, y=Y, sw=1.2)
txt("ro_d", 612, Y - 62, "d gev", size=10.5, color=SUB, align="center", w=60)
txt("ro_c", 612, Y - 16, "c gev", size=10.5, color=SUB, align="center", w=60)
txt("ro_g", 612, Y + 30, "g gev", size=10.5, color=SUB, align="center", w=60)

# ---------------------------------------------------------------- fail-soft gate
rect("gate", 712, Y - 58, 110, 116, BOX_F, BOX_S, sw=1.3)
bars("gate_b", 742, Y - 18, [40, 40], w=12, gap=12, fill=GREY_F, stroke=GREY_S)
txt("gate_t", 712, Y + 18, "Gate", size=11, align="center", w=110)
arrow("a_ctx_gate", [[0, 0], [0, -60]], x=752, y=Y + 122, dash=True, color=YEL_S, sw=1.2)
txt("ctx_l", 722, Y + 126, "ctx", size=11, color=YEL_S, align="center", w=60)
arrow("a_T_gate", [[0, 0], [0, -60]], x=786, y=Y + 122, color=SUB, sw=1.1)
txt("T_l", 756, Y + 126, "d, c", size=11, color=SUB, align="center", w=60)
circ("sig", 866, Y, 14, "σ", size=13)
arrow("a_gate_sig", [[0, 0], [28, 0]], x=824, y=Y)
txt("w_l", 880, Y - 30, "w", size=12)

# ---------------------------------------------------------------- blend into the tile disparity
circ("mul", 934, Y, 14, "⊗", size=13)
arrow("a_sig_mul", [[0, 0], [38, 0]], x=882, y=Y)
circ("sub", 934, Y - 84, 14, "−", size=14)
arrow("a_sub_mul", [[0, 0], [0, 56]], x=934, y=Y - 70)
# d_gev tapped off the top read-out into the subtractor
jdot("j_dgev", 698, Y - 46)
arrow("a_dgev_sub", [[0, 0], [0, -72], [236, -72], [236, -52]], x=698, y=Y - 46, sw=1.2)
circ("add", 1000, Y, 14, "⊕", size=14)
arrow("a_mul_add", [[0, 0], [38, 0]], x=948, y=Y)
# tile disparity d enters the adder from below and the subtractor from the right
arrow("a_d_add", [[0, 0], [0, -60]], x=1000, y=Y + 78, color=SUB, sw=1.1)
jdot("j_d", 1000, Y + 50)
arrow("a_d_sub", [[0, 0], [30, 0], [30, -134], [-52, -134]], x=1000, y=Y + 50, color=SUB, sw=1.1)
txt("d_in_l", 980, Y + 82, "d", size=12, align="center", w=40)
arrow("a_out", [[0, 0], [50, 0]], x=1014, y=Y)
txt("out_l", 1070, Y - 9, "d′", size=13)

ELEMENTS = E
