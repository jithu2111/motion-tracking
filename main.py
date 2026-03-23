"""
Optical Flow, Motion Tracking, and Bilinear Interpolation

Outputs (written to ./output/):
    video1_sparse_optical_flow.mp4   — Lucas-Kanade sparse flow on video 1
    video2_sparse_optical_flow.mp4   — Lucas-Kanade sparse flow on video 2
    video1_dense_optical_flow.mp4    — Farneback dense flow (HSV) on video 1
    video2_dense_optical_flow.mp4    — Farneback dense flow (HSV) on video 2
    flow_analysis_Video1_720p.png    — Quantitative evidence for optical flow inferences
    flow_analysis_Video2_4K.png      — Quantitative evidence for optical flow inferences
    frame_validation_Video1_720p.png — Pixel-level tracking cross-validation for video 1
    frame_validation_Video2_4K.png   — Pixel-level tracking cross-validation for video 2
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

    Cross-validation approach
    -------------------------
    To avoid circular validation (where flow_uv = p1 - p0 trivially gives
    p_hat = p1), we use TWO independent methods:

      Method A — Lucas-Kanade (sparse):  gives tracked positions p1_LK
      Method B — Farneback   (dense) :  gives flow field (u, v) at every pixel

    We sample the dense Farneback flow at each feature location p0 to get
    an independent (u_F, v_F), then predict:
        p_hat = p0 + (u_F, v_F)

    The residual ||p_hat - p1_LK|| measures how closely the two independent
    methods agree.  A small residual validates that the tracking math is
    consistent across different solvers.

    We also compare intensities:
        I_bilinear  = bilinear interpolation of frame t+1 at p_hat (sub-pixel)
        I_actual    = frame t+1 at rounded p1_LK (integer pixel)
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

    # Method A — LK sparse tracking: frame0 -> frame1
    p1, st, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **LK_PARAMS)

    good0 = p0[st == 1]   # shape (N, 2)
    good1 = p1[st == 1]

    # Method B — Farneback dense flow (independent of LK)
    dense_flow = cv2.calcOpticalFlowFarneback(
        gray0, gray1, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    # Sample dense flow at each feature point to get independent (u_F, v_F)
    dense_uv = np.zeros_like(good0)
    for idx in range(len(good0)):
        xi = int(np.clip(round(good0[idx, 0]), 0, gray0.shape[1] - 1))
        yi = int(np.clip(round(good0[idx, 1]), 0, gray0.shape[0] - 1))
        dense_uv[idx, 0] = dense_flow[yi, xi, 0]   # u from Farneback
        dense_uv[idx, 1] = dense_flow[yi, xi, 1]   # v from Farneback

    # Predicted position using Farneback flow (independent of LK)
    predicted = good0 + dense_uv

    # LK flow for comparison in the table
    lk_uv = good1 - good0

    N = min(len(good0), 10)   # show first 10 points in the table
    rows = []
    for i in range(N):
        x0_f, y0_f = good0[i]
        x1_f, y1_f = good1[i]              # LK result (ground truth)
        u_f, v_f   = dense_uv[i]           # Farneback flow (independent)

        x_pred, y_pred = x0_f + u_f, y0_f + v_f   # Farneback-predicted position

        # Intensity at predicted sub-pixel location via bilinear interpolation
        I_bilinear = bilinear_interpolate(gray1, x_pred, y_pred)

        # Actual intensity at nearest integer pixel in frame1 (at LK position)
        xi = int(np.clip(round(x1_f), 0, gray1.shape[1] - 1))
        yi = int(np.clip(round(y1_f), 0, gray1.shape[0] - 1))
        I_actual = int(gray1[yi, xi])

        # Residual: distance between Farneback prediction and LK tracked position
        residual = np.sqrt((x_pred - x1_f)**2 + (y_pred - y1_f)**2)
        rows.append((i, x0_f, y0_f, x1_f, y1_f,
                     x_pred, y_pred, u_f, v_f,
                     int(I_bilinear), I_actual, residual))

    # -----------------------------------------------------------------------
    # Plot: two frames + tracked points + validation table
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"Tracking Validation — {label}\n"
                 "Cross-validation: Farneback (dense) predicted positions vs "
                 "Lucas-Kanade (sparse) tracked positions",
                 fontsize=13, fontweight="bold")

    # Frame 0 with detected features
    ax0 = fig.add_subplot(2, 3, 1)
    ax0.imshow(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
    ax0.scatter(good0[:, 0], good0[:, 1], s=20, c="lime",
                edgecolors="black", linewidths=0.5, label="Features (t)")
    ax0.set_title("Frame t  — Detected Features (Shi-Tomasi)")
    ax0.legend(fontsize=7)
    ax0.axis("off")

    # Frame 1 with LK tracked + Farneback predicted points
    ax1 = fig.add_subplot(2, 3, 2)
    ax1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    ax1.scatter(good1[:, 0], good1[:, 1], s=20, c="red",
                edgecolors="black", linewidths=0.5, label="LK tracked (t+1)")
    ax1.scatter(predicted[:, 0], predicted[:, 1], s=40,
                marker="x", c="yellow", linewidths=1.5,
                label="Farneback predicted p_hat")
    ax1.set_title("Frame t+1 — LK Tracked vs Farneback Predicted")
    ax1.legend(fontsize=7)
    ax1.axis("off")

    # Flow quiver plot (using Farneback flow)
    ax2 = fig.add_subplot(2, 3, 3)
    ax2.imshow(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB), alpha=0.6)
    step = max(1, len(good0) // 50)
    ax2.quiver(good0[::step, 0], good0[::step, 1],
               dense_uv[::step, 0], dense_uv[::step, 1],
               color="yellow", scale=200, width=0.003)
    ax2.set_title("Farneback Flow Vectors (u, v)")
    ax2.axis("off")

    # Validation table
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.axis("off")
    col_labels = ["Pt", "x0", "y0",
                  "x1(LK)", "y1(LK)",
                  "x_pred(F)", "y_pred(F)",
                  "u(F)", "v(F)",
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

    # Compute summary stats for the title
    residuals = [r[11] for r in rows]
    mean_res = np.mean(residuals)
    max_res = np.max(residuals)

    ax3.set_title(
        "Pixel-Level Cross-Validation Table\n"
        "x0,y0 = feature in frame t | x1,y1 = LK tracked position in t+1 | "
        "x_pred,y_pred = Farneback predicted = (x0+u_F, y0+v_F)\n"
        "I_bilinear = interpolated intensity at Farneback predicted pos | "
        "I_actual = pixel intensity at LK pos | "
        f"Residual = ||p_hat_F - p_LK||   "
        f"(mean={mean_res:.4f}, max={max_res:.4f})",
        fontsize=9, pad=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Validation saved  -> {out_png}")


# ===========================================================================
# SECTION 5 — OPTICAL FLOW INFERENCE  (quantitative analysis)
# ===========================================================================

def print_flow_analysis(video_path, label, out_png, scale=1.0):
    """
    Compute a single dense-flow frame and produce:
      - Console: quantitative statistics
      - PNG figure with 6 subplots providing numerical/visual evidence
        for each inference that can be drawn from optical flow.
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
    h, w = g0.shape

    flow = cv2.calcOpticalFlowFarneback(g0, g1, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    ang_deg = np.degrees(ang)

    # --- Derived quantities --------------------------------------------------
    moving_mask   = mag > 0.5                         # foreground threshold
    static_mask   = ~moving_mask
    pct_moving    = moving_mask.mean() * 100
    pct_static    = static_mask.mean() * 100
    mean_mag      = mag.mean()
    max_mag       = mag.max()
    std_mag       = mag.std()

    # Dominant direction among clearly-moving pixels
    strong_mask = mag > 1.0
    if strong_mask.any():
        dom_angle  = ang_deg[strong_mask].mean()
        dir_std    = ang_deg[strong_mask].std()
    else:
        dom_angle, dir_std = 0.0, 0.0

    # Foreground vs background mean magnitude (object segmentation evidence)
    fg_mean = mag[moving_mask].mean() if moving_mask.any() else 0
    bg_mean = mag[static_mask].mean() if static_mask.any() else 0

    # Ego-motion test: if direction std is low -> coherent global motion
    ego_motion_likely = dir_std < 60

    # Depth / parallax: compare magnitude in top-half vs bottom-half
    top_mag  = mag[:h//2, :].mean()
    bot_mag  = mag[h//2:, :].mean()

    # --- Console output ------------------------------------------------------
    print(f"\n  [{label}] Optical-Flow Quantitative Evidence (frames 0->1)")
    print(f"    Image size          : {w} x {h}")
    print(f"    Mean magnitude      : {mean_mag:.3f} px/frame")
    print(f"    Max  magnitude      : {max_mag:.3f} px/frame")
    print(f"    Std  magnitude      : {std_mag:.3f} px/frame")
    print(f"    Foreground (mag>0.5): {pct_moving:.1f}%  mean={fg_mean:.3f} px/frame")
    print(f"    Background (mag<=0.5): {pct_static:.1f}%  mean={bg_mean:.3f} px/frame")
    print(f"    Dominant direction  : {dom_angle:.1f} deg  (std={dir_std:.1f} deg)")
    print(f"    Ego-motion likely   : {'Yes' if ego_motion_likely else 'No'}"
          f"  (direction std {'<' if ego_motion_likely else '>='} 60 deg)")
    print(f"    Top-half mean mag   : {top_mag:.3f}   Bottom-half: {bot_mag:.3f}"
          f"   (parallax ratio: {bot_mag/top_mag:.2f}x)" if top_mag > 0
          else f"    Top-half mean mag   : {top_mag:.3f}   Bottom-half: {bot_mag:.3f}")

    # --- Figure with 6 evidence panels ---------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(f"Optical Flow Evidence — {label}", fontsize=15, fontweight="bold")

    # (1) Magnitude histogram — object segmentation evidence
    ax = axes[0, 0]
    ax.hist(mag.ravel(), bins=100, range=(0, max_mag), color="steelblue",
            edgecolor="none", alpha=0.8)
    ax.axvline(0.5, color="red", ls="--", lw=1.5, label=f"threshold = 0.5 px")
    ax.set_xlabel("Flow Magnitude (px/frame)")
    ax.set_ylabel("Pixel Count")
    ax.set_title("1. Object Segmentation\nMagnitude Histogram")
    ax.legend(fontsize=8)
    ax.text(0.97, 0.95,
            f"Static: {pct_static:.1f}% (mean={bg_mean:.3f})\n"
            f"Moving: {pct_moving:.1f}% (mean={fg_mean:.3f})",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # (2) Thresholded magnitude map — foreground / background mask
    ax = axes[0, 1]
    seg_vis = np.zeros((h, w, 3), dtype=np.uint8)
    seg_vis[moving_mask] = [0, 255, 0]     # green  = moving
    seg_vis[static_mask] = [40, 40, 40]    # dark   = static
    ax.imshow(seg_vis)
    ax.set_title("2. Foreground vs Background\nGreen = moving | Dark = static")
    ax.axis("off")
    ax.text(0.02, 0.02,
            f"FG: {pct_moving:.1f}%  BG: {pct_static:.1f}%",
            transform=ax.transAxes, fontsize=9, color="white",
            bbox=dict(boxstyle="round", fc="black", alpha=0.6))

    # (3) Direction histogram (polar) — motion direction + ego-motion evidence
    ax = axes[0, 2]
    dir_data = ang_deg[strong_mask].ravel() if strong_mask.any() else ang_deg.ravel()
    ax.hist(dir_data, bins=72, range=(0, 360), color="coral",
            edgecolor="none", alpha=0.8)
    ax.axvline(dom_angle, color="navy", ls="--", lw=1.5,
               label=f"dominant = {dom_angle:.1f}°")
    ax.set_xlabel("Direction (degrees)")
    ax.set_ylabel("Pixel Count")
    ax.set_title("3. Motion Direction & Ego-Motion\nDirection Histogram (moving pixels)")
    ax.legend(fontsize=8)
    verdict = "Coherent (ego-motion)" if ego_motion_likely else "Scattered (independent objects)"
    ax.text(0.97, 0.95,
            f"Mean dir: {dom_angle:.1f}°\nStd dir: {dir_std:.1f}°\n→ {verdict}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # (4) HSV flow visualisation — velocity + direction evidence
    ax = axes[1, 0]
    hsv_img = np.zeros((h, w, 3), dtype=np.uint8)
    hsv_img[..., 0] = (ang_deg / 2).astype(np.uint8)        # hue  = direction
    hsv_img[..., 1] = 255                                    # full saturation
    hsv_img[..., 2] = cv2.normalize(mag, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
    flow_rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    ax.imshow(flow_rgb)
    ax.set_title("4. Velocity & Direction Map (HSV)\nBright = fast | Colour = direction")
    ax.axis("off")
    ax.text(0.02, 0.02,
            f"Mean speed: {mean_mag:.2f} px/frame\nMax speed: {max_mag:.2f} px/frame",
            transform=ax.transAxes, fontsize=9, color="white",
            bbox=dict(boxstyle="round", fc="black", alpha=0.6))

    # (5) Top-half vs bottom-half magnitude — depth / parallax evidence
    ax = axes[1, 1]
    zones = ["Top Half\n(far)", "Bottom Half\n(near)"]
    vals  = [top_mag, bot_mag]
    bars  = ax.bar(zones, vals, color=["#3498db", "#e74c3c"], width=0.5)
    ax.set_ylabel("Mean Flow Magnitude (px/frame)")
    ax.set_title("5. Depth Cue (Parallax)\nNearer objects flow faster")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ratio = bot_mag / top_mag if top_mag > 0 else 0
    ax.text(0.97, 0.95,
            f"Ratio (bottom/top): {ratio:.2f}x\n"
            f"{'Bottom faster → near objects move more' if ratio > 1 else 'Top faster or similar → flat scene / camera motion'}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # (6) Summary evidence table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["Object Segmentation",   f"{pct_moving:.1f}% moving, {pct_static:.1f}% static",
         f"FG mean={fg_mean:.3f}, BG mean={bg_mean:.3f}"],
        ["Camera Ego-Motion",     f"Dir std = {dir_std:.1f}°",
         "Yes" if ego_motion_likely else "No — independent motion"],
        ["Object Velocity",       f"Mean = {mean_mag:.3f} px/fr",
         f"Max = {max_mag:.3f} px/fr"],
        ["Motion Direction",      f"Dominant = {dom_angle:.1f}°",
         f"Spread = {dir_std:.1f}°"],
        ["Depth / Parallax",      f"Top = {top_mag:.3f}, Bot = {bot_mag:.3f}",
         f"Ratio = {ratio:.2f}x"],
        ["Event Detection",       f"Max spike = {max_mag:.3f} px/fr",
         f"Std = {std_mag:.3f} px/fr"],
    ]
    col_labels = ["Inference", "Key Metric", "Evidence"]
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    ax.set_title("6. Numerical Evidence Summary", fontsize=11, pad=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Evidence figure saved -> {out_png}")


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

        print_flow_analysis(
            vpath, label,
            os.path.join(OUTPUT_DIR, f"flow_analysis_{label}.png"),
            scale=sc)

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