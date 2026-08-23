"""Hand-laid spec for the StereoLite full-architecture figure (IGEV/RAFT idiom, terse labels)."""
OUT = "architecture"

BLUE_F, BLUE_S = "#dbeafe", "#1e40af"        # 2D feature encoder bars
YEL_F, YEL_S = "#fef3c7", "#b45309"          # context encoder bars
PINK_F, PINK_S = "#fecaca", "#b91c1c"        # 3D conv bars
ORG_F, ORG_S = "#fed7aa", "#c2410c"          # ConvGRU bars
CUBE = ("#bfdbfe", "#93c5fd", "#60a5fa", "#1e3a5f")
GCUBE = ("#a7f3d0", "#6ee7b7", "#34d399", "#047857")
BOX_F, BOX_S = "#f8fafc", "#64748b"
INK, SUB, RED = "#1e293b", "#475569", "#dc2626"

E = []


def rect(i, x, y, w, h, fill, stroke, text=None, size=12, sw=1.5, round=True, tcolor=None, dash=False):
    d = dict(kind="rect", id=i, x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw, round=round, dash=dash)
    if text:
        d.update(text=text, size=size, tcolor=tcolor or INK)
    E.append(d)


def txt(i, x, y, text, size=12, color=INK, align="left", bold=False, w=None):
    d = dict(kind="text", id=i, x=x, y=y, text=text, size=size, color=color, align=align, bold=bold)
    if w: d["w"] = w
    E.append(d)


def bars(prefix, x, yc, heights, w=13, gap=9, fill=BLUE_F, stroke=BLUE_S):
    for k, h in enumerate(heights):
        rect(f"{prefix}{k}", x + k * (w + gap), yc - h / 2, w, h, fill, stroke, round=False)


def cube(prefix, x, y, w, h, dep, cols, label=None, lsize=11):
    fr, tp, sd, st = cols
    E.append(dict(kind="poly", id=prefix + "_top", x=x, y=y - dep, fill=tp, stroke=st, sw=1.2,
                  points=[[0, dep], [dep, 0], [w + dep, 0], [w, dep]]))
    E.append(dict(kind="poly", id=prefix + "_side", x=x + w, y=y - dep, fill=sd, stroke=st, sw=1.2,
                  points=[[0, dep], [dep, 0], [dep, h], [0, h + dep]]))
    E.append(dict(kind="rect", id=prefix + "_front", x=x, y=y, w=w, h=h, fill=fr, stroke=st, sw=1.2, round=False))
    if label:
        txt(prefix + "_lab", x - 40, y + h + 8, label, size=lsize, color=INK, align="center", w=w + dep + 80)


def arrow(i, pts, x=0, y=0, color=INK, sw=1.5, dash=False, head="arrow"):
    E.append(dict(kind="arrow", id=i, x=x, y=y, points=pts, color=color, sw=sw, dash=dash, head=head))


def dot(i, x, y, r=6, fill=RED):
    E.append(dict(kind="ellipse", id=i, x=x - r, y=y - r, w=2 * r, h=2 * r, fill=fill, stroke=fill, sw=1))


def circ(i, x, y, r, text, size=12, fill="#ffffff", stroke=INK):
    E.append(dict(kind="ellipse", id=i, x=x - r, y=y - r, w=2 * r, h=2 * r, fill=fill, stroke=stroke, sw=1.3,
                  text=text, size=size, tcolor=INK))


def gru(prefix, x, y, label, scale):
    rect(prefix, x, y, 118, 104, BOX_F, ORG_S, round=True, sw=1.3)
    bars(prefix + "_b", x + 31, y + 36, [54, 36, 54], w=12, gap=10, fill=ORG_F, stroke=ORG_S)
    txt(prefix + "_t", x, y + 68, label, size=11, align="center", w=118)
    txt(prefix + "_s", x, y + 84, scale, size=9.5, color=SUB, align="center", w=118)


def plane_icon(prefix, x, y):
    E.append(dict(kind="poly", id=prefix + "_p", x=x, y=y, fill="#e2e8f0", stroke=SUB, sw=1.2,
                  points=[[0, 18], [16, 0], [44, 0], [28, 18]]))
    txt(prefix + "_t", x, y + 21, "↑2", size=10, color=SUB, align="center", w=44)


# ---------------------------------------------------------------- inputs + encoders
E.append(dict(kind="image", id="imL", x=50, y=70, w=150, h=84, asset="left"))
E.append(dict(kind="image", id="imR", x=50, y=200, w=150, h=84, asset="right"))
txt("imL_l", 50, 158, "Left Image", size=11, color=INK, align="center", w=150)
txt("imR_l", 50, 288, "Right Image", size=11, color=INK, align="center", w=150)

bars("fe", 262, 180, [180, 140, 100])
txt("fe_s1", 255, 275, "1/4", size=9, color=SUB)
txt("fe_s2", 279, 275, "1/8", size=9, color=SUB)
txt("fe_s3", 300, 275, "1/16", size=9, color=SUB)
txt("fe_l", 230, 290, "Feature Encoder", size=11, color=INK, align="center", w=120)
arrow("a_imL", [[0, 0], [50, 0]], x=203, y=112)
arrow("a_imR", [[0, 0], [50, 0]], x=203, y=242)

bars("ce", 262, 420, [70, 70], fill=YEL_F, stroke=YEL_S)
txt("ce_l", 230, 465, "Context Encoder", size=11, align="center", w=120)
arrow("a_ctx_in", [[0, 0], [-22, 0], [-22, 308], [227, 308]], x=48, y=112, dash=True, head="arrow", color=SUB)

# ---------------------------------------------------------------- cost volume + tile init
arrow("a_f16_cv", [[0, 0], [70, 0]], x=322, y=178)
cube("cv", 400, 130, 110, 90, 26, CUBE, label="Cost Volume (1/16)")
arrow("a_cv_3d", [[0, 0], [52, 0]], x=540, y=175)
rect("reg", 596, 120, 150, 112, BOX_F, PINK_S, sw=1.3)
bars("reg_b", 618, 166, [56, 76, 76, 40], w=12, gap=12, fill=PINK_F, stroke=PINK_S)
txt("reg_t", 596, 210, "3D Regularization", size=11, align="center", w=150)
arrow("a_3d_init", [[0, 0], [44, 0]], x=750, y=175)
rect("init", 798, 128, 160, 96, BOX_F, BOX_S, sw=1.3, text="Tile-State Init", size=12)
# T0 routed down and left into the first GRU block
arrow("a_T0", [[0, 0], [0, 78], [-428, 78], [-428, 230], [-393, 230]], x=878, y=226)
txt("a_T0_l", 458, 282, "T0", size=11, color=INK)

# ---------------------------------------------------------------- recurrent tile refinement chain
GX = [485, 737, 1015]
GY = 412
gru("g1", GX[0], GY, "ConvGRU ×2", "1/16")
gru("g2", GX[1], GY, "ConvGRU ×3", "1/8")
gru("g3", GX[2], GY, "ConvGRU ×3", "1/4")

# one matching-feature line from the encoder, with a drop into each lookup circle
BUSY = 336
arrow("bus", [[0, 0], [28, 0], [28, BUSY - 215], [GX[2] + 59 - 322, BUSY - 215]], x=322, y=215, color=SUB, sw=1, head=None, dash=True)
for k, gx in enumerate(GX):
    cx = gx + 59
    circ(f"L{k}", cx, 374, 13, "L", size=12)
    arrow(f"bus_drop{k}", [[0, 0], [0, 24]], x=cx, y=BUSY, color=SUB, sw=1, dash=True)
    arrow(f"a_L{k}", [[0, 0], [0, 24]], x=cx, y=388)

# between-block propagation: output dot, plane icon, arrow
YC = GY + 50
arrow("a_g1_g2", [[0, 0], [134, 0]], x=603, y=YC)
dot("d16", 621, YC)
plane_icon("pl1", 648, YC - 44)
arrow("a_g2_add", [[0, 0], [98, 0]], x=855, y=YC)
dot("d8", 873, YC)
plane_icon("pl2", 900, YC - 44)
circ("add", 966, YC, 12, "⊕", size=14)
arrow("a_add_g3", [[0, 0], [37, 0]], x=978, y=YC)

# context to every block (dashed)
arrow("a_ctx", [[0, 0], [0, 135], [790, 135]], x=300, y=457, dash=True, color=YEL_S, sw=1.2, head=None)
for k, gx in enumerate(GX):
    arrow(f"a_ctx_up{k}", [[0, 0], [0, -74]], x=gx + 59, y=592, dash=True, color=YEL_S, sw=1.2)

# ---------------------------------------------------------------- narrow-band GEV + fail-soft fusion
cube("gev", 1070, 130, 112, 90, 26, GCUBE, label="Narrow-Band GEV (1/4)")
arrow("a_f4_gev", [[0, 0], [0, -52], [858, -52], [858, 48]], x=268, y=90, dash=True, color=SUB, sw=1, head="arrow")
arrow("a_d_gev", [[0, 0], [0, -196], [76, -196], [76, -281], [122, -281]], x=944, y=YC - 6, dash=True, color=SUB, sw=1, head="arrow")
txt("a_d_l", 1026, 192, "d", size=11, color=INK)
arrow("a_gev_3d", [[0, 0], [48, 0]], x=1212, y=175)
dot("dgev", 1228, 175)
bars("g3d", 1270, 175, [60, 60, 60], w=11, gap=8, fill=PINK_F, stroke=PINK_S)
arrow("a_3d_fus", [[0, 0], [44, 0]], x=1330, y=175)
rect("fus", 1378, 128, 160, 96, BOX_F, BOX_S, sw=1.3, text="Fail-Soft Fusion", size=12)
arrow("a_fus_add", [[0, 0], [0, 78], [-492, 78], [-492, 148]], x=1458, y=224)
txt("a_fus_l", 1200, 284, "w · Δd", size=11, color=INK)

# ---------------------------------------------------------------- render + upsample + output
arrow("a_g3_up", [[0, 0], [72, 0]], x=1133, y=YC)
dot("d4", 1155, YC)
rect("up", 1209, GY, 170, 100, BOX_F, BOX_S, sw=1.3, text="Plane Rendering\n& Convex Upsampling", size=11)
arrow("a_up_out", [[0, 0], [62, 0]], x=1381, y=YC)
dot("dfull", 1401, YC)
E.append(dict(kind="image", id="imD", x=1447, y=YC - 42, w=150, h=84, asset="disp"))
txt("imD_l", 1447, YC + 46, "Final Disparity", size=11, color=INK, align="center", w=150)

# ---------------------------------------------------------------- legend
LY = 640
items = [("2D Conv", BLUE_F, BLUE_S), ("Context Conv", YEL_F, YEL_S), ("3D Conv", PINK_F, PINK_S),
         ("ConvGRU", ORG_F, ORG_S)]
lx = 30
for k, (name, f, s) in enumerate(items):
    rect(f"lg{k}", lx, LY, 14, 14, f, s, round=False, sw=1.2)
    txt(f"lg{k}_t", lx + 20, LY - 1, name, size=10.5, color=SUB)
    lx += 20 + len(name) * 6.4 + 30
circ("lgL", lx + 7, LY + 7, 8, "L", size=9)
txt("lgL_t", lx + 22, LY - 1, "Local correlation lookup", size=10.5, color=SUB)
lx += 22 + 24 * 6.4 + 30
E.append(dict(kind="poly", id="lgp", x=lx, y=LY + 2, fill="#e2e8f0", stroke=SUB, sw=1.2, points=[[0, 12], [8, 0], [24, 0], [16, 12]]))
txt("lgp_t", lx + 30, LY - 1, "Plane-aware propagation", size=10.5, color=SUB)
lx += 30 + 24 * 6.4 + 30
dot("lgdot", lx + 7, LY + 7)
txt("lgdot_t", lx + 20, LY - 1, "Supervised output", size=10.5, color=SUB)

ELEMENTS = E
