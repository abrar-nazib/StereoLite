"""Live stereo demonstration for the thesis supervisor demo.

A single OpenCV window shows a horizontally stacked frame:

    [ left camera view | disparity view (TURBO colormap) ]

It runs live StereoLite inference (the thesis checkpoint, architecture
"gev4_opt_narrow_plane") and offers three interactive features for the demo:

  1. Persistent click to depth. Click any pixel in either panel and a
     distance marker sticks there. On every following frame each stored
     marker re-reads the current disparity at its pixel and prints the
     metric depth Z = fx * baseline / disp_native in both panels. Markers
     accumulate so you can label several objects at once. Press 'c' to
     clear them all.

  2. Generate PointCloud button. A labelled rectangle sits in the top
     left corner. Click inside it (or press 'p') to snapshot the current
     left RGB plus disparity, build a coloured Open3D point cloud, save a
     .ply to /tmp, and open it in an Open3D viewer window.

  3. Standard controls: 'q' or ESC quit, 'f' freeze / resume, 's' save the
     current stacked frame PNG to /tmp.

Two input sources, selected with --source:

  * dataset (default): loops over rectified real frames already on disk in
    /media/abrar/AbrarSSD/Datasets/stereo_samples_20260425_104147/, so the
    entire UI can be demoed today with no camera attached.
  * camera: opens /dev/video2 as a 2560x720 side by side stereo stream and
    splits it into two 1280x720 eyes.

Design goal: keep everything ready so that when the Waveshare AR0144
stereo rig is plugged in, going live needs only one flag change and no
code edits:

    python model/scripts/demo_supervisor.py                # dataset frames
    python model/scripts/demo_supervisor.py --source camera # live rig

There is also a headless self test that needs neither a display nor a
camera (loads model, runs one inference, prints disparity stats and a
centre depth, builds a point cloud, writes a .ply):

    python model/scripts/demo_supervisor.py --selftest

Camera geometry follows disparity_to_pointcloud.py: Waveshare AR0144
stereo, per eye 1280x720, baseline 52 mm, horizontal FOV 65 deg, so
fx = 640 / tan(32.5 deg) is about 1005 px at the native 1280 width.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Wayland sessions break both OpenCV's Qt highgui and Open3D's GLFW+GLEW
# viewer (GLEW needs an X11/GLX context, which native Wayland does not give).
# Force the X11 / XWayland path so cv2 windows and the Open3D point-cloud
# viewer both get a working OpenGL context. Must run before importing cv2.
if os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.pop("WAYLAND_DISPLAY", None)

import cv2
import numpy as np
import torch

# --- Path + environment setup (must precede model imports) ---
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJ, "model", "scripts"))
sys.path.insert(0, os.path.join(PROJ, "model", "designs"))
os.environ.setdefault("XFORMERS_DISABLED", "1")

# build_model lives in the ablation harness; reuse it verbatim so the demo
# always instantiates the exact thesis architecture.
from overfit_efficiency_ablation import build_model  # noqa: E402
# Reuse the validated stereo geometry (disparity to XYZ + RGB) instead of
# reimplementing it here.
from disparity_to_pointcloud import disparity_to_points, write_ply  # noqa: E402

# --- Fixed resolutions ---
TRAIN_H, TRAIN_W = 384, 640      # model input (matches training)
NATIVE_W, NATIVE_H = 1280, 720   # per eye native resolution (display + depth)
DISP_SCALE = NATIVE_W / TRAIN_W  # 2.0: convert 640 wide disparity to native px

# Inference resolution (set from CLI in main). Higher = crisper edges but
# slower and noisier on textureless regions. Disparity is always rescaled to
# native pixels afterwards, so depth/cloud scale is invariant to this choice.
INF = {"w": TRAIN_W, "h": TRAIN_H}

DATASET_ROOT = "/media/abrar/AbrarSSD/Datasets/stereo_samples_20260425_104147"

# On-frame button geometry (top left of the composed window), in composed
# pixel coordinates. BTN = generate point cloud, CLR_BTN = clear depth points.
BTN = dict(x0=12, y0=12, w=220, h=44)
CLR_BTN = dict(x0=242, y0=12, w=150, h=44)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def resolve_ckpt(user_ckpt: str | None) -> str:
    """Pick the checkpoint to load.

    Preference order: an explicit --ckpt, else the fine tuned real camera
    checkpoint (produced by a separate process, may not exist yet), else
    the base thesis checkpoint that always exists.
    """
    if user_ckpt:
        return user_ckpt
    # Preference: native_crop fine-tune (protocol-matched to 960 inference),
    # then the resize fine-tune, then the base SceneFlow checkpoint.
    for rel in ("model/checkpoints/finetune_realcam_ncrop_best.pth",
                "model/checkpoints/finetune_realcam_best.pth",
                "model/benchmarks/20260704_fullsf_gev4onp_nc/best.pth"):
        p = os.path.join(PROJ, rel)
        if os.path.exists(p):
            return p
    return os.path.join(PROJ, "model/benchmarks/20260704_fullsf_gev4onp_nc/best.pth")


def load_model(ckpt_path: str, device: torch.device):
    """Instantiate gev4_opt_narrow_plane and load the checkpoint."""
    model, cfg = build_model("gev4_opt_narrow_plane")
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded checkpoint: {ckpt_path}")
    print(f"architecture: gev4_opt_narrow_plane  "
          f"({n_params / 1e6:.3f} M params)  device={device}")
    return model, cfg


@torch.no_grad()
def infer_disparity(model, L_bgr: np.ndarray, R_bgr: np.ndarray,
                    device: torch.device) -> tuple[np.ndarray, float]:
    """Run one stereo forward pass.

    Inputs are native BGR frames (any size). They are resized to
    TRAIN_W x TRAIN_H, converted BGR to RGB, fed as float in [0, 1], and
    the returned disparity is upscaled to native resolution and multiplied
    by DISP_SCALE so its values are in native pixel units.

    Returns (disp_native, latency_ms) where disp_native is (NATIVE_H,
    NATIVE_W) float32 disparity in native pixels.
    """
    L = cv2.resize(L_bgr, (INF["w"], INF["h"]), interpolation=cv2.INTER_AREA)
    R = cv2.resize(R_bgr, (INF["w"], INF["h"]), interpolation=cv2.INTER_AREA)

    def to_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Normalization convention: float in [0, 1] (divide uint8 by 255).
        t = torch.from_numpy(rgb).float() / 255.0
        return t.permute(2, 0, 1).unsqueeze(0).to(device)

    Lt, Rt = to_tensor(L), to_tensor(R)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = model(Lt, Rt, aux=True)
    disp = out["d_final"][0, 0].float().cpu().numpy()  # (INF h, INF w), INF-w units
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000.0

    # Upscale to native and rescale disparity magnitude to native pixels.
    disp_native = cv2.resize(disp, (NATIVE_W, NATIVE_H),
                             interpolation=cv2.INTER_LINEAR) * (NATIVE_W / INF["w"])
    return disp_native.astype(np.float32), ms


def depth_from_disp(disp_native_px: float, focal_px: float,
                    baseline_m: float) -> float:
    """Metric depth from a single native-pixel disparity value.

    Guard against disparities below 1 px (numerically unstable / invalid).
    """
    if disp_native_px is None or disp_native_px < 1.0 or \
            not np.isfinite(disp_native_px):
        return float("nan")
    return focal_px * baseline_m / disp_native_px


# ---------------------------------------------------------------------------
# Point cloud
# ---------------------------------------------------------------------------

# One shared offscreen renderer, created lazily and reused across clicks
# (Filament dislikes being re-instantiated within a process).
_PC_RENDERER = {"r": None}


def _show_pointcloud_interactive(pcd, win="StereoLite point cloud",
                                 width=1024, height=768):
    """Wayland-safe interactive point-cloud viewer.

    Open3D's GLFW/GLEW window cannot get a GL context on native Wayland, so
    the cloud is rendered offscreen with the EGL renderer and shown in a cv2
    window driven by an orbit camera: drag to rotate, +/- (or scroll) to
    zoom, r to reset, q or ESC to close and return to the live demo.
    """
    import open3d.visualization.rendering as rendering

    if len(pcd.points) == 0:
        print("point cloud: nothing to display")
        return

    aabb = pcd.get_axis_aligned_bounding_box()
    center = np.asarray(aabb.get_center(), dtype=np.float64)
    radius = float(np.linalg.norm(aabb.get_extent())) / 2.0 or 1.0

    if _PC_RENDERER["r"] is None:
        _PC_RENDERER["r"] = rendering.OffscreenRenderer(width, height)
    renderer = _PC_RENDERER["r"]
    renderer.scene.clear_geometry()
    renderer.scene.set_background([0.08, 0.08, 0.10, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.5
    renderer.scene.add_geometry("pc", pcd, mat)

    st = dict(az=0.0, el=0.3, dist=1.8 * radius, last=None)
    up = [0.0, -1.0, 0.0]  # image Y points down, so -Y up shows it upright.

    def eye():
        e, a, d = st["el"], st["az"], st["dist"]
        return center + np.array([d * np.cos(e) * np.sin(a),
                                  d * np.sin(e),
                                  d * np.cos(e) * np.cos(a)])

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            st["last"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            st["last"] = None
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON) \
                and st["last"] is not None:
            dx, dy = x - st["last"][0], y - st["last"][1]
            st["az"] -= dx * 0.008
            st["el"] = float(np.clip(st["el"] + dy * 0.008, -1.4, 1.4))
            st["last"] = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            st["dist"] *= 0.9 if flags > 0 else 1.1

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, width, height)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        renderer.setup_camera(60.0, center.tolist(), eye().tolist(), up)
        rgb = np.asarray(renderer.render_to_image())
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, "drag = rotate   +/- = zoom   r = reset   q = close",
                    (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, frame)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            break
        elif k in (ord("+"), ord("=")):
            st["dist"] *= 0.9
        elif k in (ord("-"), ord("_")):
            st["dist"] *= 1.1
        elif k == ord("r"):
            st.update(az=0.0, el=0.3, dist=1.8 * radius)
    cv2.destroyWindow(win)


def build_and_show_pointcloud(L_bgr: np.ndarray, disp_native: np.ndarray,
                              focal_px: float, baseline_m: float,
                              stride: int, max_depth: float,
                              show_window: bool,
                              ply_out: str) -> int:
    """Build a coloured Open3D point cloud, save a .ply, optionally view it.

    Returns the point count. When show_window is False (self test) the
    Open3D GUI is never opened.
    """
    import open3d as o3d

    L_rgb = cv2.cvtColor(L_bgr, cv2.COLOR_BGR2RGB)
    # disparity_to_points expects image and disparity at matching resolution.
    # min_disp floor of 2 px caps depth at f*B/2 (about 26 m here) so a few
    # low-confidence far pixels do not stretch the cloud over a huge Z range.
    pts, cols = disparity_to_points(
        L_rgb, disp_native, f_px=focal_px, baseline_m=baseline_m,
        min_disp=2.0, max_depth_m=max_depth, stride=stride)

    n = len(pts)
    if n == 0:
        print("point cloud: 0 valid points (disparity too small everywhere?)")
        return 0

    # Save a .ply for later inspection (plain writer from the geometry module).
    write_ply(Path(ply_out), pts, cols)
    print(f"point cloud: {n:,} points -> {ply_out}")

    if show_window:
        # Build an Open3D cloud, drop depth-edge fliers, and open the
        # Wayland-safe offscreen viewer.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print("opening point-cloud viewer (drag to rotate, +/- zoom, q closes)")
        try:
            _show_pointcloud_interactive(pcd)
        except Exception as exc:
            print(f"viewer failed to open ({exc}).")
            print(f"The cloud is saved at {ply_out}; open it with:")
            print(f"    python -c \"import open3d as o3d; "
                  f"o3d.visualization.draw_geometries("
                  f"[o3d.io.read_point_cloud('{ply_out}')])\"")
    return n


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------

class DatasetSource:
    """Loops over rectified real left/right frames already on disk.

    Yields (L_bgr, R_bgr) at native 1280x720. Loops forever with a small
    delay so the demo runs at roughly a target fps without a camera.
    """

    def __init__(self, root: str, fps: float = 10.0):
        self.root = Path(root)
        clean = self.root / "clean_pairs.txt"
        if clean.exists():
            self.bases = [ln.strip() for ln in clean.read_text().splitlines()
                          if ln.strip()]
        else:
            self.bases = sorted(p.stem for p in
                                (self.root / "left").glob("*.png"))
        if not self.bases:
            raise RuntimeError(f"no frames found under {root}")
        self.idx = 0
        self.delay = 1.0 / max(fps, 1e-3)
        print(f"dataset source: {len(self.bases)} pairs from {root}")

    def read(self):
        b = self.bases[self.idx % len(self.bases)]
        self.idx += 1
        L = cv2.imread(str(self.root / "left" / f"{b}.png"))
        R = cv2.imread(str(self.root / "right" / f"{b}.png"))
        if L is None or R is None:
            # Skip a missing pair rather than crashing the live demo.
            return self.read()
        time.sleep(self.delay)  # pace the loop like a ~10 fps feed
        return L, R

    def release(self):
        pass


class CameraSource:
    """Opens /dev/video<device> as 2560x720 side by side stereo, splits L/R.

    Reuses the MJPG open + split logic from live_stereolite.py so switching
    to the live rig is only a --source change.
    """

    def __init__(self, device: int, cam_w: int = 2560, cam_h: int = 720,
                 fps: int = 60):
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open /dev/video{device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"camera source: /dev/video{device} {aw}x{ah}")
        if aw != cam_w:
            print(f"warning: expected {cam_w}px wide, got {aw}px "
                  f"(may be a single camera crop, not stereo)")
        self.cap = cap

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("camera frame grab failed")
        mid = frame.shape[1] // 2
        L, R = frame[:, :mid], frame[:, mid:]
        # Ensure native per eye resolution for consistent geometry.
        if L.shape[1] != NATIVE_W or L.shape[0] != NATIVE_H:
            L = cv2.resize(L, (NATIVE_W, NATIVE_H))
            R = cv2.resize(R, (NATIVE_W, NATIVE_H))
        return L, R

    def release(self):
        self.cap.release()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def colourise(disp: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)


def draw_marker(img: np.ndarray, x: int, y: int, label: str,
                colour=(255, 255, 255)):
    """Draw a small circle + a readable text label at (x, y)."""
    cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
    cv2.circle(img, (x, y), 4, colour, -1)
    org = (x + 8, max(y - 8, 14))
    cv2.putText(img, label, (org[0] + 1, org[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, org,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


def draw_button(img: np.ndarray):
    """Draw the on-frame buttons (Generate PointCloud + Clear Points)."""
    x0, y0, w, h = BTN["x0"], BTN["y0"], BTN["w"], BTN["h"]
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), -1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (0, 200, 255), 2)
    cv2.putText(img, "Generate PointCloud", (x0 + 10, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    cx0, cy0, cw, ch = CLR_BTN["x0"], CLR_BTN["y0"], CLR_BTN["w"], CLR_BTN["h"]
    cv2.rectangle(img, (cx0, cy0), (cx0 + cw, cy0 + ch), (40, 40, 40), -1)
    cv2.rectangle(img, (cx0, cy0), (cx0 + cw, cy0 + ch), (80, 80, 255), 2)
    cv2.putText(img, "Clear Points", (cx0 + 10, cy0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 255), 1, cv2.LINE_AA)


def _inside(mx: int, my: int, b: dict) -> bool:
    return b["x0"] <= mx <= b["x0"] + b["w"] and b["y0"] <= my <= b["y0"] + b["h"]


def inside_button(mx: int, my: int) -> bool:
    return _inside(mx, my, BTN)


def inside_clear(mx: int, my: int) -> bool:
    return _inside(mx, my, CLR_BTN)


def annotate_bar(img: np.ndarray, lines: list[str]):
    """Draw a black info bar with white text at the bottom of a panel."""
    h = 22
    bar_h = h * len(lines) + 6
    y_top = img.shape[0] - bar_h
    cv2.rectangle(img, (0, y_top), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (8, y_top + (i + 1) * h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Self test (headless, no window, no camera)
# ---------------------------------------------------------------------------

def load_liteanystereo(ckpt_path: str, device: torch.device):
    """Load the official LiteAnyStereo (7.6 M, foundation-era lightweight).

    Uses the vendored repo under model/scripts/modal/lite_any_stereo_repo.
    """
    las_root = os.path.join(PROJ, "model", "scripts", "modal",
                            "lite_any_stereo_repo")
    if las_root not in sys.path:
        sys.path.insert(0, las_root)
    from core.liteanystereo import LiteAnyStereo
    model = LiteAnyStereo()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    inc = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded checkpoint: {ckpt_path}")
    print(f"architecture: LiteAnyStereo  ({n / 1e6:.3f} M params, "
          f"{len(inc.missing_keys)} missing)  device={device}")
    return model


@torch.no_grad()
def infer_disparity_las(model, L_bgr: np.ndarray, R_bgr: np.ndarray,
                        device: torch.device) -> tuple[np.ndarray, float]:
    """LiteAnyStereo forward. Same 384x640 protocol as infer_disparity, but
    LiteAnyStereo takes RGB in [0, 255] (no /255) and needs /32 padding."""
    from core.utils.utils import InputPadder
    L = cv2.resize(L_bgr, (INF["w"], INF["h"]), interpolation=cv2.INTER_AREA)
    R = cv2.resize(R_bgr, (INF["w"], INF["h"]), interpolation=cv2.INTER_AREA)

    def to_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

    Lt, Rt = to_tensor(L), to_tensor(R)
    padder = InputPadder(Lt.shape, divis_by=32)
    Ltp, Rtp = padder.pad(Lt, Rt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    disp = model(Ltp, Rtp, max_disp=192, test_mode=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000.0
    disp = padder.unpad(disp).squeeze().float().cpu().numpy()  # (INF h, INF w)
    disp_native = cv2.resize(disp, (NATIVE_W, NATIVE_H),
                             interpolation=cv2.INTER_LINEAR) * (NATIVE_W / INF["w"])
    return disp_native.astype(np.float32), ms


def make_sgbm():
    """Classical OpenCV StereoSGBM, run at native resolution (CPU). A grounded
    non-learned baseline for the live A/B."""
    nd = 192  # must be divisible by 16; covers up to 192 px disparity
    bs = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=bs,
        P1=8 * 3 * bs * bs, P2=32 * 3 * bs * bs, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    print("classical StereoSGBM  (numDisparities=192, blockSize=5, native res)")
    return matcher


def infer_disparity_sgbm(matcher, L_bgr, R_bgr, device):
    """SGBM on native-resolution grayscale. Disparity is already in native px."""
    gl = cv2.cvtColor(L_bgr, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(R_bgr, cv2.COLOR_BGR2GRAY)
    t0 = time.time()
    disp16 = matcher.compute(gl, gr)          # 16 * disparity, int16
    ms = (time.time() - t0) * 1000.0
    disp = disp16.astype(np.float32) / 16.0
    disp[disp < 0] = 0.0                       # invalid pixels -> 0
    return disp.astype(np.float32), ms


def build_runtime(args, device):
    """Return (model, infer_fn) for the requested --model."""
    if args.model == "sgbm":
        return make_sgbm(), infer_disparity_sgbm
    if args.model == "liteanystereo":
        ckpt = args.ckpt or os.path.join(PROJ, "model", "checkpoints",
                                         "LiteAnyStereo.pth")
        return load_liteanystereo(ckpt, device), infer_disparity_las
    model, _ = load_model(resolve_ckpt(args.ckpt), device)
    return model, infer_disparity


def run_selftest(args, device):
    print("=== demo_supervisor self test (headless) ===")
    model, infer_fn = build_runtime(args, device)

    src = DatasetSource(args.dataset_root, fps=1000.0)
    L, R = src.read()
    print(f"frame: left {L.shape}  right {R.shape}")

    disp, ms = infer_fn(model, L, R, device)
    valid = disp[disp > 1.0]
    print(f"disparity: shape {disp.shape}  latency {ms:.1f} ms")
    if valid.size:
        print(f"disparity (valid > 1 px): min {valid.min():.2f}  "
              f"max {valid.max():.2f}  median {np.median(valid):.2f}  "
              f"(native pixels)")
    else:
        print("disparity: no valid pixels above 1 px")

    cy, cx = NATIVE_H // 2, NATIVE_W // 2
    d_c = float(disp[cy, cx])
    z_c = depth_from_disp(d_c, args.focal, args.baseline)
    print(f"centre pixel ({cx}, {cy}): disp {d_c:.2f} px  depth {z_c:.3f} m")

    ply = "/tmp/demo_supervisor_selftest.ply"
    n = build_and_show_pointcloud(
        L, disp, focal_px=args.focal, baseline_m=args.baseline,
        stride=args.stride, max_depth=args.max_depth,
        show_window=False, ply_out=ply)
    print(f"point cloud: {n:,} points written to {ply}")
    print("=== self test OK ===")
    src.release()


# ---------------------------------------------------------------------------
# Live demo
# ---------------------------------------------------------------------------

def run_live(args, device):
    model, infer_fn = build_runtime(args, device)

    if args.source == "camera":
        src = CameraSource(args.device)
    else:
        src = DatasetSource(args.dataset_root, fps=args.fps)

    # Warmup so the first displayed latency is representative.
    warm_L, warm_R = src.read()
    for _ in range(3):
        infer_fn(model, warm_L, warm_R, device)

    win = "StereoLite supervisor demo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1600, 450)

    # Mutable UI state shared with the mouse callback.
    state = dict(
        points=[],            # list of (x_img, y_img) in native left-panel px
        panel_w=NATIVE_W,     # left-panel width in composed coords
        disp=None,            # current native disparity (for click lookups)
        L=None,               # current left BGR (for point cloud snapshot)
        trigger_pc=False,     # set True when the button is clicked
        display_scale=1.0,    # composed->window scale (for mapping clicks back)
    )

    def on_mouse(event, mx, my, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Map the window click back to composed-image coordinates.
        cx = int(mx / max(state["display_scale"], 1e-6))
        cy = int(my / max(state["display_scale"], 1e-6))
        # 1) Buttons take priority over depth-point placement.
        if inside_button(cx, cy):
            state["trigger_pc"] = True
            return
        if inside_clear(cx, cy):
            state["points"].clear()
            print("cleared depth markers (button)")
            return
        # 2) Otherwise treat as a depth point. A click in the right panel
        #    maps to the same image pixel as the left panel (subtract the
        #    left-panel width offset).
        pw = state["panel_w"]
        img_x = cx - pw if cx >= pw else cx
        img_y = cy
        img_x = int(np.clip(img_x, 0, NATIVE_W - 1))
        img_y = int(np.clip(img_y, 0, NATIVE_H - 1))
        state["points"].append((img_x, img_y))

    cv2.setMouseCallback(win, on_mouse)

    print("controls: q/ESC quit  f freeze  s save  c clear points  "
          "p point cloud  (or click the button)")

    frozen = None
    save_count = 0
    pc_count = 0
    ms_hist: list[float] = []

    while True:
        if frozen is None:
            L, R = src.read()
        else:
            L, R = frozen

        disp, ms = infer_fn(model, L, R, device)
        ms_hist.append(ms)
        ms_hist = ms_hist[-30:]
        med_ms = float(np.median(ms_hist))

        # Colour map bounds from the valid disparity percentiles.
        valid = disp[disp > 1.0]
        if valid.size > 64:
            lo = float(np.percentile(valid, 5))
            hi = float(np.percentile(valid, 95))
        else:
            lo, hi = 0.0, 96.0
        disp_col = colourise(disp, lo, hi)

        state["disp"] = disp
        state["L"] = L
        state["panel_w"] = L.shape[1]

        left_panel = L.copy()
        right_panel = disp_col

        # Draw every persistent depth marker in BOTH panels.
        for i, (px, py) in enumerate(state["points"]):
            d_here = float(disp[py, px])
            z = depth_from_disp(d_here, args.focal, args.baseline)
            label = f"{i + 1}: {z:.2f} m" if np.isfinite(z) else f"{i + 1}: n/a"
            draw_marker(left_panel, px, py, label, (0, 255, 0))
            draw_marker(right_panel, px, py, label, (255, 255, 255))

        annotate_bar(left_panel, [
            f"left  |  source={args.source}  {NATIVE_W}x{NATIVE_H}",
            f"markers={len(state['points'])}  (click to add, c to clear)"
            + ("  FROZEN" if frozen is not None else ""),
        ])
        annotate_bar(right_panel, [
            f"disparity (TURBO)  {med_ms:.0f} ms  inf {INF['w']}x{INF['h']}"
            f"  range {lo:.1f}..{hi:.1f} px",
            "click a pixel for depth  |  p or button = point cloud",
        ])

        composed = np.hstack([left_panel, right_panel])
        draw_button(composed)

        # Fit the composed frame to the window width and remember the scale
        # so mouse clicks map back to composed coordinates.
        target_w = 1600
        scale = min(1.0, target_w / composed.shape[1])
        state["display_scale"] = scale
        if scale < 1.0:
            disp_frame = cv2.resize(
                composed, (int(composed.shape[1] * scale),
                           int(composed.shape[0] * scale)))
        else:
            disp_frame = composed
        cv2.imshow(win, disp_frame)

        # Point cloud trigger (button click sets the flag).
        if state["trigger_pc"]:
            state["trigger_pc"] = False
            ply = f"/tmp/demo_supervisor_pc_{pc_count:03d}.ply"
            build_and_show_pointcloud(
                L, disp, focal_px=args.focal, baseline_m=args.baseline,
                stride=args.stride, max_depth=args.max_depth,
                show_window=True, ply_out=ply)
            pc_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == ord("f"):
            frozen = (L.copy(), R.copy()) if frozen is None else None
            print("frozen" if frozen is not None else "resumed")
        elif key == ord("s"):
            out = f"/tmp/demo_supervisor_frame_{save_count:03d}.png"
            cv2.imwrite(out, composed)
            print(f"saved {out}")
            save_count += 1
        elif key == ord("c"):
            state["points"].clear()
            print("cleared depth markers")
        elif key == ord("p"):
            ply = f"/tmp/demo_supervisor_pc_{pc_count:03d}.ply"
            build_and_show_pointcloud(
                L, disp, focal_px=args.focal, baseline_m=args.baseline,
                stride=args.stride, max_depth=args.max_depth,
                show_window=True, ply_out=ply)
            pc_count += 1

    src.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Live StereoLite supervisor demo (dataset or camera).")
    p.add_argument("--source", choices=["dataset", "camera"],
                   default="dataset",
                   help="dataset = looped disk frames (works now); "
                        "camera = live rig on /dev/video<device>")
    p.add_argument("--device", type=int, default=2,
                   help="/dev/video<N> for the stereo camera (source=camera)")
    p.add_argument("--dataset_root", default=DATASET_ROOT,
                   help="root with left/, right/, clean_pairs.txt")
    p.add_argument("--model", choices=["stereolite", "liteanystereo", "sgbm"],
                   default="stereolite",
                   help="stereolite = our fine-tuned edge model (default); "
                        "liteanystereo = 7.6 M foundation-era reference; "
                        "sgbm = classical OpenCV StereoSGBM baseline")
    p.add_argument("--ckpt", default=None,
                   help="checkpoint path; stereolite default prefers "
                        "finetune_realcam_best.pth then base best.pth; "
                        "liteanystereo default is checkpoints/LiteAnyStereo.pth")
    p.add_argument("--focal", type=float, default=1005.0,
                   help="horizontal focal length in px at native 1280 width")
    p.add_argument("--baseline", type=float, default=0.052,
                   help="stereo baseline in metres (default AR0144 = 52 mm)")
    p.add_argument("--stride", type=int, default=2,
                   help="point-cloud pixel stride (2 = about 25%% of points)")
    p.add_argument("--max_depth", type=float, default=20.0,
                   help="discard point-cloud points beyond this many metres")
    p.add_argument("--fps", type=float, default=10.0,
                   help="dataset source playback rate")
    p.add_argument("--inf_width", type=int, default=0,
                   help="inference width (0 = auto: 1280 for stereolite, 640 "
                        "for liteanystereo). Higher = crisper but slower.")
    p.add_argument("--inf_height", type=int, default=0,
                   help="inference height (0 = auto, paired with inf_width)")
    p.add_argument("--selftest", action="store_true",
                   help="headless self test (no window, no camera)")
    args = p.parse_args()

    # Resolve inference resolution (rounded to a multiple of 16 for the model).
    if args.inf_width and args.inf_height:
        iw, ih = args.inf_width, args.inf_height
    elif args.model == "liteanystereo":
        iw, ih = 640, 384          # LiteAnyStereo is heavy; keep it low
    else:
        # Match the SceneFlow training density (native_crop at 960x540 native
        # sampling). Feeding the camera at 960 wide keeps disparities in the
        # trained range: 1280 doubles them (out of range -> speckle) and 640
        # halves them (the resize protocol that oversmooths).
        iw, ih = 960, 544
    INF["w"] = max(16, (iw // 16) * 16)
    INF["h"] = max(16, (ih // 16) * 16)
    print(f"inference resolution: {INF['w']}x{INF['h']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.selftest:
        run_selftest(args, device)
    else:
        run_live(args, device)


if __name__ == "__main__":
    main()
