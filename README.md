# Motion Tracking — Optical Flow, Bilinear Interpolation & Tracking Validation

---

## How to Run

### 1. Clone / open the repository

```bash
git clone <repo-url>
cd "motion-tracking"
```

### 2. Place your videos

Put the two source videos inside a `Videos/` folder at the project root:

```
Videos/
  7334251-hd_1920_1080_25fps.mp4    # Video 1 — 1920x1080, 38 s
  4887663-hd_2048_1080_25fps.mp4    # Video 2 — 2048x1080, 53 s
```

### 3. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the script

```bash
python main.py
```

### 5. Outputs

All results are written to `./output/`:

| File | Description |
|-----|-------------|
| `video1_sparse_optical_flow.mp4` | Lucas-Kanade sparse flow — Video 1 |
| `video2_sparse_optical_flow.mp4` | Lucas-Kanade sparse flow — Video 2 |
| `video1_dense_optical_flow.mp4`  | Farneback dense HSV flow — Video 1 |
| `video2_dense_optical_flow.mp4`  | Farneback dense HSV flow — Video 2 |
| `flow_analysis_Video1_720p.png`  | Quantitative evidence figure (6 panels) — Video 1 |
| `flow_analysis_Video2_4K.png`    | Quantitative evidence figure (6 panels) — Video 2 |
| `frame_validation_Video1_720p.png`  | LK vs Farneback cross-validation — Video 1 |
| `frame_validation_Video2_4K.png`    | LK vs Farneback cross-validation — Video 2 |

---

## Concepts Used

---

### 1. Optical Flow

Optical flow is the apparent motion of pixels between two consecutive frames caused by the movement of objects or the camera. It is represented as a 2D vector field where each vector `(u, v)` at pixel `(x, y)` describes how far that pixel has moved in the next frame.

**Brightness Constancy Assumption:**

The foundation of optical flow is that the intensity of a pixel does not change as it moves:

```
I(x, y, t) = I(x + u, y + v, t + 1)
```

Taylor-expanding the right-hand side for small displacements and applying the constraint gives the **Optical Flow Constraint Equation (OFCE)**:

```
Ix*u + Iy*v + It = 0
```

where `Ix`, `Iy` are spatial image gradients and `It` is the temporal gradient. This is one equation with two unknowns — additional constraints are needed to solve it.

---

### 2. Sparse Optical Flow — Lucas-Kanade Method

The **Lucas-Kanade** method adds a **spatial coherence assumption**: all pixels within a local window `W` share the same flow vector `(u, v)`. This gives an over-determined system:

```
A * d = b

A = [Ix1  Iy1]     d = [u]     b = [-It1]
    [Ix2  Iy2]         [v]         [-It2]
    [ ...  ...]                    [ ...]
```

Solved via **least squares (normal equations)**:

```
(A^T A) d = A^T b
d = (A^T A)^-1 * A^T b
```

Where:

```
A^T A = [ sum(Ix^2)    sum(Ix*Iy) ]
        [ sum(Ix*Iy)   sum(Iy^2)  ]

A^T b = [ -sum(Ix*It) ]
        [ -sum(Iy*It) ]
```

The matrix `A^T A` must be invertible — this requires the patch to have texture in at least two distinct gradient directions (i.e., a corner, not a flat region or edge). This is why **Shi-Tomasi corner detection** is used to select trackable feature points.

**Pyramidal LK** extends this to handle larger motions by computing flow at multiple image scales (coarse to fine).

**What sparse flow tells us:**
- Which feature points are moving and in what direction
- Local object velocity at tracked keypoints
- Whether the camera or scene objects are in motion

---

### 3. Dense Optical Flow — Farneback Method

The **Farneback** method computes a flow vector for **every pixel** in the frame by approximating the local neighbourhood of each pixel using a polynomial expansion, then deriving the displacement from how that polynomial changes between frames.

**HSV Visualisation:**
- **Hue** → direction of motion (0°–360° mapped to colour wheel)
- **Saturation** → fixed at maximum (255)
- **Value (brightness)** → magnitude of motion, normalised to [0, 255]

This means:
- **Bright regions** = fast-moving areas
- **Dark regions** = static background
- **Colour** = direction of travel

**What dense flow tells us:**
- Full per-pixel motion field — superior to sparse for segmentation
- Camera ego-motion (pan/tilt creates a global directional pattern)
- Moving object boundaries via magnitude discontinuities
- Motion parallax: pixels closer to the camera flow faster than distant ones
- Dominant motion direction of the scene

---

### 4. Motion Tracking Equations — Derivation from Fundamentals

Given the OFCE solution `(u, v)` at a feature point `p0 = (x0, y0)` in frame `t`, the **predicted position** in frame `t+1` is:

```
p_hat = p0 + (u, v) = (x0 + u, y0 + v)
```

This is derived from the discrete version of the continuous motion equation:

```
dx/dt = u,   dy/dt = v
=> x(t+1) = x(t) + u * dt,  dt = 1 frame
```

**Tracking problem setup for two image frames:**

Given frame `I_t` and frame `I_{t+1}`:
1. Detect feature points `{p_i}` in `I_t` using Shi-Tomasi
2. For each `p_i`, solve the LK normal equations to obtain `(u_i, v_i)`
3. Predict next position: `p_hat_i = p_i + (u_i, v_i)`
4. Verify: compare `p_hat_i` against LK-reported `p1_i` — residual should be ~0
5. Use bilinear interpolation to evaluate `I_{t+1}(p_hat_i)` and compare with the actual pixel intensity

---

### 5. Bilinear Interpolation — Derivation

When a predicted position `(x, y)` is sub-pixel (non-integer), the intensity cannot be read directly from the pixel grid. **Bilinear interpolation** estimates it from the four surrounding integer pixels.

Let:
```
x0 = floor(x),   x1 = x0 + 1,   dx = x - x0
y0 = floor(y),   y1 = y0 + 1,   dy = y - y0
```

The four surrounding intensities are:
```
Q11 = I(x0, y0)    Q21 = I(x1, y0)   <- top row
Q12 = I(x0, y1)    Q22 = I(x1, y1)   <- bottom row
```

**Step 1 — interpolate horizontally:**
```
R1 = (1 - dx) * Q11 + dx * Q21    (along top edge)
R2 = (1 - dx) * Q12 + dx * Q22    (along bottom edge)
```

**Step 2 — interpolate vertically:**
```
P = (1 - dy) * R1 + dy * R2
```

**Expanded form:**
```
P = (1-dx)(1-dy)*Q11 + dx(1-dy)*Q21
  + (1-dx)*dy*Q12    + dx*dy*Q22
```

All four weights are non-negative and sum to 1 — this is a **convex combination**, guaranteeing the interpolated value stays within the valid intensity range. It produces smooth, continuous intensity estimates with no ringing artefacts.

---

### 6. Tracking Validation (Cross-Validation)

To validate the tracking math, two **independent** optical flow methods are compared:

- **Method A — Lucas-Kanade (sparse):** tracks feature points directly, giving positions `p1_LK`
- **Method B — Farneback (dense):** computes a full flow field `(u_F, v_F)` at every pixel

The Farneback flow is sampled at each detected feature location `p0` to predict:
```
p_hat = p0 + (u_F, v_F)
```

The **residual** `||p_hat - p1_LK||` measures how closely these two independent methods agree. A small residual confirms that the tracking equations produce consistent results regardless of the solver used.

For each of two consecutive frames from both videos, the program:

1. Detects Shi-Tomasi corners in frame `t`
2. Runs Lucas-Kanade to get tracked positions `p1_LK` in frame `t+1`
3. Runs Farneback dense flow independently to get `(u_F, v_F)` at each feature
4. Computes predicted position: `p_hat = p0 + (u_F, v_F)`
5. Computes residual: `||p_hat - p1_LK||` (small = methods agree)
6. Uses bilinear interpolation to read intensity at `p_hat` in frame `t+1` and compares with the actual pixel intensity at `p1_LK`

The validation PNG for each video shows:
- **Frame t** with detected features
- **Frame t+1** with LK-tracked (red) vs Farneback-predicted (yellow ×) positions
- **Quiver plot** of Farneback flow vectors
- **Table** of 10 sample points with all coordinates, intensities, and residuals

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | >= 4.x | Optical flow, video I/O, feature detection |
| `numpy` | >= 1.x | Matrix operations, array math |
| `matplotlib` | >= 3.x | Validation plots and tables |


