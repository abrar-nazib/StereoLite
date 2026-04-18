import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = 2          # /dev/video2 — CCB Camera (stereo)
FRAME_W     = 2560       # Full side-by-side width
FRAME_H     = 720
FPS         = 60         # MJPG supports 60fps at 2560x720
# ─────────────────────────────────────────────────────────────────────────────

def open_stereo_camera():
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{DEVICE}")

    cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,          FPS)

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Opened /dev/video{DEVICE}: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")

    if actual_w != FRAME_W:
        print(f"WARNING: Expected {FRAME_W}px wide, got {actual_w}px — "
              f"you may be getting a single-camera crop instead of stereo!")
    return cap


def split_stereo(frame):
    """Split side-by-side stereo frame into left and right."""
    mid = frame.shape[1] // 2
    return frame[:, :mid], frame[:, mid:]


def main():
    cap = open_stereo_camera()
    save_count = 0

    print("Controls: 'q' quit | 's' save frame pair | 'f' toggle full/split view")
    show_full = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed — check USB connection.")
            break

        left, right = split_stereo(frame)

        if show_full:
            cv2.imshow("Stereo Raw (2560x720)", frame)
        else:
            # Side-by-side with a visible divider
            divider = np.full((FRAME_H, 3, 3), (0, 255, 0), dtype=np.uint8)
            preview = np.hstack([left, divider, right])
            cv2.imshow("Left | Right", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"left_{save_count:04d}.png",  left)
            cv2.imwrite(f"right_{save_count:04d}.png", right)
            print(f"Saved pair {save_count:04d}  →  left_{save_count:04d}.png / right_{save_count:04d}.png")
            save_count += 1
        elif key == ord('f'):
            show_full = not show_full
            cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()