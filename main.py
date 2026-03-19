"""
Optical Flow, Motion Tracking, and Bilinear Interpolation

Outputs (written to ./output/):
    video1_sparse_optical_flow.mp4   — Lucas-Kanade sparse flow on video 1
    video2_sparse_optical_flow.mp4   — Lucas-Kanade sparse flow on video 2
    video1_dense_optical_flow.mp4    — Farneback dense flow (HSV) on video 1
    video2_dense_optical_flow.mp4    — Farneback dense flow (HSV) on video 2
    frame_validation_video1.png      — Pixel-level tracking validation for video 1
    frame_validation_video2.png      — Pixel-level tracking validation for video 2
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_PATHS = [
    "Videos/7334251-hd_1920_1080_25fps.mp4",   # video 1  (1920x1080, 38 s)
    "Videos/4887663-hd_2048_1080_25fps.mp4",   # video 2  (2048x1080, 53 s)
]
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Downscale factor for video 2 so processing stays fast
SCALE_4K = 0.5           # 2048x1080 -> 1024x540

# Lucas-Kanade / Shi-Tomasi parameters
FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01,
                      minDistance=7, blockSize=7)
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                            30, 0.01))

# Colours for drawing tracks
TRACK_COLOR  = (0, 255,   0)   # green  — active track
POINT_COLOR  = (0,   0, 255)   # red    — current point
ARROW_COLOR  = (255, 200,  0)  # gold   — motion arrow


# ===========================================================================
# SECTION 1 — SPARSE OPTICAL FLOW  (Lucas-Kanade)
# ===========================================================================

def compute_sparse_optical_flow(video_path, out_path, scale=1.0):
    """
    Sparse optical flow using the Lucas-Kanade method.

    The Lucas-Kanade method assumes:
      1. Brightness constancy  : I(x, y, t) = I(x+u, y+v, t+1)
      2. Small motion          : Taylor-expand and linearise
      3. Spatial coherence     : same (u,v) over a local window W

    The optical-flow constraint equation is:
        Ix*u + Iy*v + It = 0          (one equation, two unknowns)

    Over a window of n pixels we get the over-determined system A*d = b :
        A = [Ix1 Iy1]   d = [u]   b = [-It1]
            [Ix2 Iy2]       [v]       [-It2]
            [  ...   ]               [  ... ]

    Least-squares solution (normal equations):
        (A^T A) d = A^T b
        d = (A^T A)^-1 A^T b

    Where A^T A = [sum(Ix^2)   sum(IxIy)]   A^T b = [-sum(IxIt)]
                  [sum(IxIy)   sum(Iy^2) ]           [-sum(IyIt)]
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, first = cap.read()
    if not ret:
        print(f"  [!] Cannot open {video_path}")
        return

    if scale != 1.0:
        first = cv2.resize(first, None, fx=scale, fy=scale)

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
    mask = np.zeros_like(first)          # canvas for persistent trails

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is None or len(p0) < 10:
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
            mask = np.zeros_like(frame)

        if p0 is not None:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0,
                                                   None, **LK_PARAMS)
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for new, old in zip(good_new, good_old):
                a, b_  = new.ravel().astype(int)
                c, d_  = old.ravel().astype(int)
                mask = cv2.line(mask,  (a, b_), (c, d_), TRACK_COLOR, 1)
                frame = cv2.arrowedLine(frame, (c, d_), (a, b_),
                                        ARROW_COLOR, 1, tipLength=0.3)
                frame = cv2.circle(frame, (a, b_), 3, POINT_COLOR, -1)

            p0 = good_new.reshape(-1, 1, 2)

        out_frame = cv2.add(frame, mask)
        cv2.putText(out_frame, "Sparse Optical Flow (Lucas-Kanade)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(out_frame, f"Frame {frame_idx:04d}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)

        writer.write(out_frame)
        prev_gray = gray.copy()
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  Sparse flow saved -> {out_path}  ({frame_idx} frames)")


# ===========================================================================
# SECTION 2 — DENSE OPTICAL FLOW  (Farneback + HSV heatmap)
# ===========================================================================

def compute_dense_optical_flow(video_path, out_path, scale=1.0):
    """
    Dense optical flow using the Gunnar Farneback method.

    Every pixel gets a flow vector (u, v).  We map these to HSV colour:
        Hue        -> direction of motion  (angle of the flow vector)
        Saturation -> 255  (fully saturated)
        Value      -> magnitude (speed), normalised to [0, 255]

    This produces an intuitive visualisation:
        Bright pixels  = fast-moving regions
        Hue colour     = direction of travel
        Dark pixels    = static background
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, first = cap.read()
    if not ret:
        return
    if scale != 1.0:
        first = cv2.resize(first, None, fx=scale, fy=scale)

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255          # saturation always max

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2          # hue  = direction
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255,
                                    cv2.NORM_MINMAX)    # value = speed

        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Side-by-side: original | flow heatmap
        combo = np.hstack([frame, flow_bgr])
        combo = cv2.resize(combo, (w, h))   # keep output same size

        cv2.putText(combo, "Original",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(combo, "Dense Optical Flow (HSV heatmap)",
                    (w // 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(combo, f"Frame {frame_idx:04d}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1)

        writer.write(combo)
        prev_gray = gray.copy()
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  Dense flow saved  -> {out_path}  ({frame_idx} frames)")


# ===========================================================================
# SECTION 3 — BILINEAR INTERPOLATION  (from first principles)
# ===========================================================================

def bilinear_interpolate(image, x, y):
    """
    Bilinear Interpolation — derivation
    ====================================
    Given a sub-pixel location (x, y) we need to estimate the intensity.

    Let:
        x0, y0 = floor(x), floor(y)   (top-left integer neighbour)
        x1, y1 = x0+1,    y0+1        (bottom-right integer neighbour)
        dx = x - x0,  dy = y - y0     (fractional parts, in [0,1))

    The four surrounding pixel values are:
        Q11 = I(x0, y0),  Q21 = I(x1, y0)   <- top row
        Q12 = I(x0, y1),  Q22 = I(x1, y1)   <- bottom row

    Step 1 — interpolate along x (horizontal):
        R1 = (1-dx)*Q11 + dx*Q21     (top edge)
        R2 = (1-dx)*Q12 + dx*Q22     (bottom edge)

    Step 2 — interpolate along y (vertical):
        P  = (1-dy)*R1  + dy*R2

    Expanding fully:
        P = (1-dx)(1-dy)*Q11 + dx(1-dy)*Q21
          +  (1-dx)*dy  *Q12 + dx*dy   *Q22

    Each term is a bilinear weight times the corresponding corner value.
    The weights sum to 1, so the interpolation is a convex combination.
    """
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    dx, dy = x - x0, y - y0

    h, w = image.shape[:2]
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    Q11 = image[y0, x0].astype(float)
    Q21 = image[y0, x1].astype(float)
    Q12 = image[y1, x0].astype(float)
    Q22 = image[y1, x1].astype(float)

    R1 = (1 - dx) * Q11 + dx * Q21
    R2 = (1 - dx) * Q12 + dx * Q22
    P  = (1 - dy) * R1  + dy * R2
    return np.round(P).astype(np.uint8)


# ===========================================================================
# SECTION 4 — TRACKING VALIDATION  (theoretical vs actual pixel locations)
# ===========================================================================

def validate_tracking(video_path, label, out_png, scale=1.0):
    """
    Motion Tracking Equations — derivation
    =======================================
    The brightness constancy assumption gives:
        I(x, y, t) = I(x + u*dt, y + v*dt, t + dt)

    Taylor-expand the right-hand side (small motion, dt = 1 frame):
        I(x+u, y+v, t+1) ~ I(x,y,t) + Ix*u + Iy*v + It

    Brightness constancy => LHS = RHS:
        Ix*u + Iy*v + It = 0       <- Optical-Flow Constraint Equation (OFCE)

    For a feature point tracked from frame t to t+1:
        predicted position:  p_hat = p + (u, v)

    (u, v) comes from the LK solution described in Section 1.

    Validation procedure
    --------------------
    1. Detect feature corners in frame t   (Shi-Tomasi)
    2. Run LK to get flow vectors (u, v)
    3. Compute predicted position:  p_hat = p0 + (u, v)
    4. Compare p_hat vs p1 (LK output)  — residual = ||p_hat - p1||
    5. Use bilinear interpolation to read intensity at p_hat in frame t+1
       and compare with actual intensity at the nearest integer pixel.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    if not ret:
        return
    ret, frame1 = cap.read()
    if not ret:
        return
    cap.release()

    if scale != 1.0:
        frame0 = cv2.resize(frame0, None, fx=scale, fy=scale)
        frame1 = cv2.resize(frame1, None, fx=scale, fy=scale)

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Detect features in frame 0
    p0 = cv2.goodFeaturesToTrack(gray0, mask=None, **FEATURE_PARAMS)
    if p0 is None:
        print(f"  [!] No features found in {video_path}")
        return

    # LK forward tracking: frame0 -> frame1
    p1, st, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **LK_PARAMS)

    good0 = p0[st == 1]   # shape (N, 2)
    good1 = p1[st == 1]

    # Flow vectors
    flow_uv = good1 - good0                    # (N, 2)

    # Theoretical (predicted) position = p0 + (u, v)
    predicted = good0 + flow_uv

    N = min(len(good0), 10)   # show first 10 points in the table
    rows = []
    for i in range(N):
        x0_f, y0_f = good0[i]
        x1_f, y1_f = good1[i]
        u, v = flow_uv[i]

        x_pred, y_pred = x0_f + u, y0_f + v

        # Intensity at predicted sub-pixel location via bilinear interpolation
        I_bilinear = bilinear_interpolate(gray1, x_pred, y_pred)

        # Actual intensity at nearest integer pixel in frame1
        xi = int(np.clip(round(x1_f), 0, gray1.shape[1] - 1))
        yi = int(np.clip(round(y1_f), 0, gray1.shape[0] - 1))
        I_actual = int(gray1[yi, xi])

        residual = np.sqrt((x_pred - x1_f)**2 + (y_pred - y1_f)**2)
        rows.append((i, x0_f, y0_f, x1_f, y1_f,
                     x_pred, y_pred, u, v,
                     int(I_bilinear), I_actual, residual))

    # -----------------------------------------------------------------------
    # Plot: two frames + tracked points + validation table
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"Tracking Validation — {label}", fontsize=14, fontweight="bold")

    # Frame 0 with detected features
    ax0 = fig.add_subplot(2, 3, 1)
    ax0.imshow(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
    ax0.scatter(good0[:, 0], good0[:, 1], s=20, c="lime",
                edgecolors="black", linewidths=0.5, label="Features (t)")
    ax0.set_title("Frame t  — Detected Features (Shi-Tomasi)")
    ax0.legend(fontsize=7)
    ax0.axis("off")

    # Frame 1 with tracked + predicted points
    ax1 = fig.add_subplot(2, 3, 2)
    ax1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    ax1.scatter(good1[:, 0], good1[:, 1], s=20, c="red",
                edgecolors="black", linewidths=0.5, label="Tracked LK (t+1)")
    ax1.scatter(predicted[:, 0], predicted[:, 1], s=40,
                marker="x", c="yellow", linewidths=1.5, label="Predicted p_hat")
    ax1.set_title("Frame t+1 — LK Tracked vs Predicted")
    ax1.legend(fontsize=7)
    ax1.axis("off")

    # Flow quiver plot
    ax2 = fig.add_subplot(2, 3, 3)
    ax2.imshow(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB), alpha=0.6)
    step = max(1, len(good0) // 50)
    ax2.quiver(good0[::step, 0], good0[::step, 1],
               flow_uv[::step, 0], flow_uv[::step, 1],
               color="yellow", scale=200, width=0.003)
    ax2.set_title("Flow Vectors (u, v)")
    ax2.axis("off")

    # Validation table
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.axis("off")
    col_labels = ["Pt", "x0", "y0",
                  "x1(LK)", "y1(LK)",
                  "x_pred", "y_pred",
                  "u", "v",
                  "I_bilinear", "I_actual", "Residual"]
    table_data = [
        [str(r[0]),
         f"{r[1]:.1f}", f"{r[2]:.1f}",
         f"{r[3]:.1f}", f"{r[4]:.1f}",
         f"{r[5]:.1f}", f"{r[6]:.1f}",
         f"{r[7]:.2f}", f"{r[8]:.2f}",
         str(r[9]), str(r[10]),
         f"{r[11]:.4f}"]
        for r in rows
    ]
    tbl = ax3.table(cellText=table_data, colLabels=col_labels,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)

    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    ax3.set_title(
        "Pixel-Level Validation Table\n"
        "x0,y0 = feature in frame t | x1,y1 = LK tracked position in frame t+1\n"
        "x_pred,y_pred = predicted = (x0+u, y0+v) | "
        "I_bilinear = interpolated intensity at predicted pos | "
        "I_actual = pixel intensity at rounded (x1,y1) | Residual = ||p_hat - p1||",
        fontsize=9, pad=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Validation saved  -> {out_png}")


# ===========================================================================
# SECTION 5 — OPTICAL FLOW INFERENCE  (quantitative analysis)
# ===========================================================================

def print_flow_analysis(video_path, label, scale=1.0):
    """
    Compute a single dense-flow frame and print quantitative statistics
    that support the interpretation of what optical flow tells us.
    """
    cap = cv2.VideoCapture(video_path)
    ret, f0 = cap.read()
    ret2, f1 = cap.read()
    cap.release()
    if not ret or not ret2:
        return

    if scale != 1.0:
        f0 = cv2.resize(f0, None, fx=scale, fy=scale)
        f1 = cv2.resize(f1, None, fx=scale, fy=scale)

    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(g0, g1, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    print(f"\n  [{label}] Optical-Flow Analysis (frames 0->1)")
    print(f"    Mean magnitude   : {mag.mean():.3f} px/frame")
    print(f"    Max  magnitude   : {mag.max():.3f} px/frame")
    print(f"    Std  magnitude   : {mag.std():.3f} px/frame")
    print(f"    Pixels moving    : {(mag > 0.5).mean()*100:.1f}%")
    dom_angle = np.degrees(ang[mag > 1.0].mean()) if (mag > 1.0).any() else 0
    print(f"    Dominant direction: {dom_angle:.1f} deg")
    print()
    print("  What optical flow tells us:")
    print("    Magnitude map  -> which regions are moving (foreground vs background)")
    print("    Direction map  -> direction of object or camera motion")
    print("    High-mag blobs -> independently moving objects (people, vehicles, ...)")
    print("    Near-zero mag  -> static background")
    print("    Global pattern -> camera ego-motion (pan / tilt / zoom)")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    labels = ["Video1_720p", "Video2_4K"]
    scales = [1.0, SCALE_4K]

    for i, (vpath, label, sc) in enumerate(zip(VIDEO_PATHS, labels, scales), 1):
        print(f"\n{'='*60}")
        print(f"  Processing {label}  ({vpath})")
        print(f"{'='*60}")

        print_flow_analysis(vpath, label, scale=sc)

        print(f"\n  [1/3] Sparse optical flow ...")
        compute_sparse_optical_flow(
            vpath,
            os.path.join(OUTPUT_DIR, f"video{i}_sparse_optical_flow.mp4"),
            scale=sc)

        print(f"  [2/3] Dense optical flow ...")
        compute_dense_optical_flow(
            vpath,
            os.path.join(OUTPUT_DIR, f"video{i}_dense_optical_flow.mp4"),
            scale=sc)

        print(f"  [3/3] Tracking validation ...")
        validate_tracking(
            vpath, label,
            os.path.join(OUTPUT_DIR, f"frame_validation_{label}.png"),
            scale=sc)

    print(f"\n{'='*60}")
    print("  All outputs written to ./output/")
    print("  -------------------------------------------------------")
    print("""
  OPTICAL FLOW — WHAT CAN BE INFERRED
  =====================================
  1. Object segmentation   : moving objects show high-magnitude flow
                             vs near-zero background.
  2. Camera ego-motion     : a global affine flow pattern indicates
                             camera pan / tilt / zoom.
  3. Object velocity       : magnitude (px/frame) directly encodes speed.
  4. Motion direction      : HSV hue encodes direction continuously.
  5. Depth cue (parallax)  : faster flow generally means closer to camera.
  6. Event detection       : sudden large-magnitude spikes signal events.

  LUCAS-KANADE DERIVATION SUMMARY
  =================================
  Brightness constancy: I(x,y,t) = I(x+u, y+v, t+1)
  Taylor-expand RHS and apply constancy:
      Ix*u + Iy*v + It = 0          (OFCE — one eq., two unknowns)
  Over a window W of n pixels:
      A = [Ix_i  Iy_i],  d = [u; v],  b = [-It_i]
  Over-determined system -> normal equations:
      (A^T A) d = A^T b
      d = (A^T A)^-1 A^T b
  A^T A invertible <=> two distinct gradient directions (textured patch).
  Predicted position: p_hat = p0 + d = (x0+u, y0+v)

  BILINEAR INTERPOLATION SUMMARY
  ================================
  For sub-pixel (x,y), let x0=floor(x), dx=x-x0, similarly y0,dy:
      R1 = (1-dx)*I(x0,y0) + dx*I(x1,y0)   [top edge]
      R2 = (1-dx)*I(x0,y1) + dx*I(x1,y1)   [bottom edge]
      P  = (1-dy)*R1 + dy*R2                [vertical blend]
  All weights >= 0, sum = 1 -> convex combination, smooth, no ringing.
  Used to evaluate image intensity at the sub-pixel predicted location.
""")
    print("  Done.")


if __name__ == "__main__":
    main()