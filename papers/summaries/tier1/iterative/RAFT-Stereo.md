# RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching

**Authors:** Lahav Lipson, Zachary Teed, Jia Deng (Princeton University)
**Venue:** 3DV 2021
**Priority:** 10/10 — THE paradigm shift of the 2020s
**Code:** https://github.com/princeton-vl/RAFT-Stereo

---

## Core Problem & Motivation

By 2021, stereo matching was dominated by 3D cost-volume architectures (GC-Net, PSMNet, GA-Net, GwcNet). These methods:
- Build a 4D cost volume of shape $H \times W \times D \times F$ (height, width, disparity, feature channels).
- Process it with **3D convolutions**, which are extremely expensive in memory and compute.
- Bake a **predefined maximum disparity** $D$ into the architecture — change the scene scale and the model can't reach the new disparities.
- Generalise poorly across datasets (trained on Scene Flow, fail on Middlebury's high-res images).

Meanwhile, optical flow had just shifted to **iterative refinement** with RAFT (Teed and Deng, ECCV 2020). The insight: instead of regressing flow once from a huge cost volume, iteratively refine an estimate by repeatedly looking up correlation values from a much cheaper all-pairs correlation volume.

**RAFT-Stereo's contribution:** adapt RAFT's iterative paradigm to stereo, exploiting the **1D disparity constraint** (matched pixels lie on the same row, so we only search horizontally) for a much cheaper cost volume, plus a multi-level GRU for global context propagation.

---

## Architecture Overview

![Figure 1: RAFT-Stereo architecture. A feature encoder (blue) extracts features from both images, used to build the correlation pyramid. A context encoder (separate CNN, applied to left image only) produces both context features (white) and an initial hidden state. The disparity field starts at 0. At each iteration, the GRU (green) samples from the correlation pyramid at the current disparity estimate, then uses the sampled correlation, image features, and current hidden state to predict a disparity update.](../../../figures/RAFTStereo_fig1_architecture.png)

Three components:
1. **Feature extraction** — two CNNs that produce per-pixel descriptors.
2. **Correlation pyramid** — pre-computed similarity table between every left-right pixel pair on each row, plus coarser versions of it.
3. **Iterative GRU update operator** — refines the disparity field one small step at a time.

The whole thing is wrapped in a loop that runs $N=22$ times during training and $N=32$ times during inference.

---

## Background primer — what is a GRU, and why do we need one here?

This is the part most stereo papers assume you already know. Skim if you do; read carefully if you don't.

### Recurrent neural networks in one paragraph
A **recurrent neural network (RNN)** is a network that processes inputs *one step at a time* while keeping a small piece of memory (a "hidden state") that carries information from one step to the next. The classic use case is reading a sentence one word at a time — each word updates an internal summary of what's been read so far. In RAFT-Stereo we're not reading a sentence; we're refining a disparity estimate. The "step" is one refinement iteration, not one word, and the "memory" is what the network has learned about the scene so far.

### Why we want memory across iterations
Imagine you predict an initial disparity, then look at the correlation values around it, then try to improve it. The next iteration looks at *new* correlation values (around the updated disparity). Without memory, the network has no way to know "I already tried moving this pixel left and it got worse" — every iteration starts from scratch. A hidden state lets the network accumulate evidence: "I keep seeing the same correlation pattern in this region, so I'm pretty confident now."

### The "gated" part
A vanilla RNN just overwrites its hidden state at every step, which makes it forget things too aggressively (or, in the opposite failure mode, makes gradients explode during training). A **gated** RNN learns *how much* to overwrite. Two famous gated variants exist: LSTM (1997, 4 gates) and **GRU** (2014, 2 gates). RAFT picks GRU because it's lighter and slightly faster while being almost as expressive.

A GRU has two gates, each a small learned function that outputs a value in $[0, 1]$ for every position:
- **Update gate $z$** — "how much of my old hidden state should I keep, vs how much should I replace with the new candidate?" $z = 0$ means keep everything old; $z = 1$ means replace everything.
- **Reset gate $r$** — "when computing the new candidate hidden state, how much should my old hidden state contribute?" $r = 0$ means ignore old state when proposing the candidate; $r = 1$ means use it fully.

Concrete intuition: the update gate is "should I change my mind?", the reset gate is "should I forget my prior reasoning when forming a new opinion?".

### Convolutional GRU vs ordinary GRU
A standard GRU treats its hidden state as a flat vector — fine for a sentence (one timestep = one word's vector). For an image, the hidden state is a **feature map** of shape $C \times H \times W$. A standard GRU would have to flatten it, losing all spatial structure. A **convolutional GRU** replaces the gate-computing matrix multiplications with **2D convolutions**, so the gates respect spatial neighbourhoods (a pixel's update gate is computed from a 3×3 region around it, not from the whole image at once). This is what RAFT-Stereo uses.

### What the hidden state represents in RAFT-Stereo
A learned, per-pixel summary of "what the network knows so far about this region's matching": correlation patterns it has seen at the current disparity, how confident it is, what the local geometry looks like. The next iteration's update is conditioned on this state plus the new lookup. After 22-32 iterations, the network has effectively run a small message-passing routine over each pixel.

---

## 1. Feature Extraction

Two CNNs process the input images. Both are residual-block stacks; both are trained from scratch.

### Feature encoder (shared between L and R)
Same weights applied to both images, producing 256-channel feature maps at 1/4 resolution (or 1/8 for the efficient variant). Outputs:
- $\mathbf{f}^l \in \mathbb{R}^{256 \times H/4 \times W/4}$ from the left image.
- $\mathbf{f}^r \in \mathbb{R}^{256 \times H/4 \times W/4}$ from the right image.

These are the **matching features** — they're what the correlation volume is built from. Sharing weights between L and R is essential: a left-image patch and its corresponding right-image patch should have nearly identical feature vectors so their dot product is high.

### Context encoder (left image only)
**Identical architecture, different weights, only run on the left image.** Produces two things:
1. **Initial hidden states** $h_0$ at each multi-level-GRU resolution (1/4, 1/8, 1/16). These seed the GRU at iteration 0.
2. **Context features** $\mathbf{c}$ — a per-pixel feature map that's injected into every GRU update step (i.e., it stays constant across iterations).

### Why two separate encoders?
The matching features need to be *interchangeable* between L and R (so dot products mean "these patches look alike"). The context features need to capture *what kind of region this is* (textureless wall vs sharp edge vs slanted ground) so the GRU can adapt its update rule. These are different jobs: one is symmetric across views, the other is single-view semantic. Two encoders, two specialisations.

For our edge tier we collapse these into one (no context encoder, matching features double as context). RAFT pays the extra params for the cleaner separation.

---

## 2. Correlation Pyramid

### The 1D correlation insight
Stereo pairs are **rectified**: the camera setup guarantees that a pixel at $(i, j)$ in the left image, if visible, appears somewhere on row $i$ in the right image — never on a different row. So when we ask "what does pixel $(i, j)$ in the left match to in the right?", we only search the left-right axis on the same row, not the entire 2D image. This collapses the search from 2D (optical flow) to 1D (stereo).

### The all-pairs 1D correlation volume
For every row, compare every left column to every right column via dot product:

$$\mathbf{C}_{ijk} = \sum_h \mathbf{f}^l_{hij} \cdot \mathbf{f}^r_{hik}, \quad \mathbf{C} \in \mathbb{R}^{H/4 \times W/4 \times W/4} \quad \text{(1)}$$

Each variable, slowly:
- $\mathbf{C}_{ijk}$ — one cell of the volume. Reads as: "the similarity between left-pixel-(row $i$, col $j$) and right-pixel-(row $i$, col $k$)".
- $i$ — row index. Same for both pixels because they're rectified to the same row.
- $j$ — column index in the left image (the pixel we're trying to match).
- $k$ — column index in the right image (a candidate match).
- $\mathbf{f}^l_{hij}$ — channel $h$ of the left feature vector at row $i$, col $j$.
- $\mathbf{f}^r_{hik}$ — channel $h$ of the right feature vector at row $i$, col $k$.
- $\sum_h$ — sum across all 256 feature channels: this is the dot product, a scalar similarity score.

For a 384×640 input, the feature maps are 96×160, and $\mathbf{C}$ has shape $96 \times 160 \times 160$ ≈ 2.5 M values. This is computed in **a single batched matrix multiplication**, not 3D convolutions.

### Why "all-pairs"
Conventional 3D-cost-volume methods only compute correlations up to a fixed max disparity (say 192 px), then squeeze the volume through 3D convs. RAFT computes correlations at *every* disparity, including 0, including the maximum possible (the image width itself). The cost is fine because it's just dot products, no convolutions on the volume.

### The pyramid (4 levels)
$\mathbf{C}^1$ is the raw volume. We then pool it along the disparity axis (the last axis, $k$) by 1D average pooling with kernel/stride 2, producing $\mathbf{C}^2$. Repeat to get $\mathbf{C}^3$ and $\mathbf{C}^4$. Each level has the **same spatial resolution** but progressively coarser **disparity resolution**.

| Level | Spatial | Disparity bins per left pixel | Each bin spans |
|---|---|---|---|
| $\mathbf{C}^1$ | 96 × 160 | 160 | 1 px |
| $\mathbf{C}^2$ | 96 × 160 | 80  | 2 px |
| $\mathbf{C}^3$ | 96 × 160 | 40  | 4 px |
| $\mathbf{C}^4$ | 96 × 160 | 20  | 8 px |

This isn't a multi-scale image pyramid (those pool space); it's a **disparity pyramid** that pools the matching range. The benefit shows up in the lookup step.

---

## 3. Iterative GRU Update Operator — deep dive

The core loop. Disparity starts at $\mathbf{d}_0 = \mathbf{0}$ everywhere. For $n = 1 \dots N$, run three steps.

### Step A — Correlation lookup
Given the current disparity $\mathbf{d}_n$, look up correlation values from the pyramid *around* the current estimate:

![Figure 2: Lookup from the correlation pyramid. For each pixel at its current disparity estimate, values are sampled from each pyramid level at integer offsets within radius r=4, using linear interpolation for non-integer positions. Retrieved values from all pyramid levels are concatenated.](../../../figures/RAFTStereo_fig2_pyramid_lookup.png)

For pixel $(i, j)$ with current disparity $d_n(i,j)$, sample at positions $d_n(i,j) + r$ for $r \in \{-4, -3, ..., +3, +4\}$ on each pyramid level. Nine offsets × four levels = **36 correlation values per pixel per iteration**.

The clever part: pyramid level 4 has bins that span 8 px each, so its 9 offsets cover ±32 px around the current disparity. Level 1 has 1-px bins, so its 9 offsets give fine-grained ±4 px detail. Combining gives "I have detailed information close to my current guess and coarse information far from it" — exactly what an iterative refinement needs.

Disparities are real-valued (not integer); positions between bins are resolved by **linear interpolation** between neighbouring bin values. So the lookup is differentiable, and the network can learn to predict sub-pixel disparities.

### Step B — GRU update (the equations)

The 36 correlation values, the current disparity, and the constant context features get folded into a 2D **convolutional GRU**:

$$x_n = [\,\text{Enc}_{\text{corr}}(\text{lookup}(\mathbf{C}, \mathbf{d}_n)),\; \text{Enc}_{\text{disp}}(\mathbf{d}_n),\; \mathbf{c}\,]$$

$$z_n = \sigma(\text{Conv}([h_{n-1}, x_n], W_z))$$

$$r_n = \sigma(\text{Conv}([h_{n-1}, x_n], W_r))$$

$$\bar{h}_n = \tanh(\text{Conv}([r_n \odot h_{n-1}, x_n], W_h))$$

$$h_n = (1 - z_n) \odot h_{n-1} + z_n \odot \bar{h}_n$$

Variable by variable:

| Symbol | What it is | Shape |
|---|---|---|
| $x_n$ | The "input" to this GRU step. Concatenation of three things: the correlation features encoded by 2 conv layers; the current disparity encoded by 2 conv layers; and the constant context features. | $C_x \times H/4 \times W/4$ |
| $h_{n-1}$ | Previous hidden state. The GRU's memory. | $128 \times H/4 \times W/4$ |
| $\mathbf{c}$ | Context features. Constant across iterations. | fixed |
| $W_z, W_r, W_h$ | Learned 3×3 convolution weights, one set per gate. | small kernels |
| $z_n$ | **Update gate** values at every pixel, in $[0,1]$ via sigmoid. "How much new vs how much old?" | $128 \times H/4 \times W/4$ |
| $r_n$ | **Reset gate** values at every pixel, in $[0,1]$ via sigmoid. "When proposing a new candidate state, how much of the old state should leak in?" | same |
| $\bar{h}_n$ | Candidate hidden state. The "what the new state would be if I fully accepted it" proposal. tanh keeps it in $[-1, 1]$. | same |
| $h_n$ | The actual new hidden state. A blend, controlled per-pixel by $z_n$. | same |
| $\sigma$ | Sigmoid — squashes any real number into $[0, 1]$. Used for the gates because they need to be mixing weights. |  |
| $\tanh$ | Hyperbolic tangent — squashes any real number into $[-1, 1]$. Used for the candidate state to keep activations bounded. |  |
| $\odot$ | **Hadamard product** — element-wise multiplication. Different from matrix multiply. |  |

### Walking through one iteration in plain English
1. *Look up* the correlation values around your current disparity guess. Encode them.
2. Encode your current disparity guess.
3. Stick them next to your context features. Call the result $x_n$.
4. Decide, per pixel, **how much to keep** of your old hidden state vs how much to replace ($z_n$).
5. Decide, per pixel, **how much your old state should influence** the candidate replacement ($r_n$).
6. Form the candidate replacement: feed [old state, modulated by $r_n$] together with $x_n$ through a conv, squash with tanh.
7. Mix old state and candidate using the update gate: $h_n = (1 - z_n) \odot h_{n-1} + z_n \odot \bar{h}_n$.
8. Pass $h_n$ through 2 small conv layers to produce a per-pixel **disparity increment** $\Delta \mathbf{d}$.

### Step C — Disparity update
$$\mathbf{d}_{n+1} = \mathbf{d}_n + \Delta \mathbf{d}$$

The update is **residual** — the network only predicts a small correction, not the full disparity. This is critical: predicting a small delta is much easier than predicting absolute disparity from scratch every iteration, and it lets the same GRU weights work at every iteration without retraining.

### What persists, what resets
- **Persists across iterations**: the hidden state $h_n$ (the GRU's memory) and the disparity field $\mathbf{d}_n$ (the running estimate).
- **Constant across iterations**: the correlation pyramid $\mathbf{C}$ (built once), the context features $\mathbf{c}$ (built once).
- **Recomputed each iteration**: the lookup (depends on current $\mathbf{d}_n$), the gates, the candidate state.

---

## Multi-Level GRU (key innovation over RAFT for flow)

A single-resolution GRU's receptive field grows by only ~3 pixels per iteration (the conv kernel size). For a 640-pixel-wide image, information from one edge wouldn't reach the other edge until iteration ~210. Way too slow for textureless regions like skies or walls, where the local correlation gives no useful signal and the network has to "borrow" disparity estimates from far away.

**Solution:** run three GRUs in parallel at three resolutions of the matching features (1/4, 1/8, 1/16 of the input image), with information flowing between them.

![Figure 3: Multi-level GRU. Three convolutional GRUs run at progressively coarser resolutions. Information passes between adjacent levels via upsampling and downsampling. Only the highest-resolution GRU (1/4) performs the correlation lookup and predicts the disparity update.](../../../figures/RAFTStereo_fig3_multilevel_gru.png)

At each iteration:
- All three GRU levels update their hidden states.
- Adjacent levels exchange information via upsampling (coarse → fine) and downsampling (fine → coarse).
- Only the **highest-resolution GRU** performs the correlation lookup and emits the disparity increment.
- The coarser GRUs provide **global context** — their effective receptive fields, in original-image terms, are much larger.

### Why this matters
At iteration $n$, the 1/16 GRU has seen "context" from $\sim 16 \times 3 \times n$ pixels of the original image. After 5 iterations that's already 240 pixels of context — enough for a wall-sized region to start propagating its disparity estimate from the textured edges into the middle. Without the multi-level structure, you'd need $\sim 50$ iterations of the 1/4 GRU to cover the same range.

### Why only the finest level decodes
The highest-resolution GRU has the most precise spatial information, so it's the one allowed to make the per-pixel decision. The coarser levels are pure "advisors" — they whisper context into the fine level via upsampling, but they don't get to commit to a disparity themselves.

---

## Slow-Fast GRU (efficiency variant)

A GRU update at 1/4 resolution costs ~16× more FLOPs than at 1/16 (2D scaling × spatial pixel count). The "fast" variant exploits this:
- Update the 1/8 and 1/16 GRUs **several times per outer iteration**.
- Update the 1/4 GRU **only once** per outer iteration.

This lets the coarse GRUs propagate context aggressively while keeping the expensive fine-resolution work cheap. On KITTI at 32 iterations, runtime drops from **0.132 s to 0.05 s** — 2.6× speedup with a small accuracy hit. This is what enables RAFT-Stereo to compete with real-time methods.

---

## Disparity Upsampling — deep dive

The GRU lives at 1/4 resolution. The output disparity has to be at full resolution. The naive answer is **bilinear upsample** — average the 4 nearest neighbours. This is *terrible* for stereo because:
- Bilinear smooths everything uniformly. Sharp depth edges (a chair's back vs the wall behind it) get blurred.
- Sub-pixel disparity precision is lost — bilinear can't put an edge between pixels.
- Disocclusion regions (where one camera sees behind an object that the other can't) get garbage values bled in from the foreground.

RAFT introduced a learned alternative called **convex upsampling** — sometimes called "convex-mask upsample" or "learned upsample mask". It's the single most effective upsample trick in modern stereo and flow.

### What "convex combination" means
A **convex combination** of values $v_1, \dots, v_9$ is a weighted average $\sum w_i v_i$ where:
- All weights are non-negative ($w_i \geq 0$).
- All weights sum to one ($\sum w_i = 1$).

The result is always inside the "convex hull" of the inputs — meaning it's somewhere in the range spanned by the values, never extrapolating outside them. For disparity, this is what we want: don't invent disparities that none of the neighbours have.

### The 9-neighbourhood
For each fine-resolution pixel, look at the 3×3 patch of 9 coarse-resolution neighbours that surround it. The fine pixel's disparity is going to be a convex combination of those 9 disparity values.

A 2× upsample turns one coarse pixel into a 2×2 block of 4 fine pixels. Each of those 4 fine pixels gets its **own** set of 9 weights — they pull from the same 9 coarse neighbours but blend them differently. So an upsample by 2 produces $4 \times 9 = 36$ weights per coarse pixel.

### How the weights are predicted
A small CNN (the **mask network**) takes the highest-resolution GRU's hidden state as input and outputs the 36 weights per coarse pixel. Importantly:
- The weights are passed through **softmax along the 9-neighbour axis** — guarantees they're non-negative and sum to one, satisfying the convex-combination constraint.
- The weights are **per-pixel and content-aware** — at a depth edge, the network can put almost all the weight on neighbours from one side of the edge, effectively propagating the foreground disparity to one half of the fine pixels and the background disparity to the other half. Bilinear has no such ability.

### Walking through one fine pixel
Suppose we're upsampling a coarse 1/4-resolution disparity to 1/2-resolution (2× upsample). Pick a coarse pixel at $(i, j)$ with disparity 12.3 px. Its 3×3 neighbourhood has the disparities:

```
11.8  12.0  12.2
12.1  12.3  12.5
12.4  12.6  12.7
```

For this coarse pixel, the mask network outputs 36 weights, organised as 4 sets of 9 (one set per fine sub-pixel). Suppose the upper-left sub-pixel's 9 weights are:

```
0.40  0.30  0.05
0.10  0.10  0.02
0.02  0.01  0.00
```
(non-negative, sum to 1)

Then that upper-left fine pixel's disparity is:

$$0.40 \times 11.8 + 0.30 \times 12.0 + 0.05 \times 12.2 + 0.10 \times 12.1 + 0.10 \times 12.3 + 0.02 \times 12.5 + 0.02 \times 12.4 + 0.01 \times 12.6 + 0.00 \times 12.7 \approx 11.96 \text{ px}$$

The upper-left fine pixel is closer to its upper-left coarse neighbours, so the network learned to weight those higher — preserving the spatial gradient cleanly. At a sharp edge, the weights on the "wrong" side would collapse to near zero and a clean step would emerge.

### In PyTorch terms
Implementation uses `F.unfold` to extract every 3×3 patch from the coarse disparity at every position, then a single `(weights * patches).sum()` produces the fine output. The full operation is roughly:

```python
# disp: (B, 1, H, W), feat: (B, C, H, W)
mask = mask_net(feat).view(B, 1, 9, scale, scale, H, W).softmax(dim=2)  # (B,1,9,2,2,H,W)
patches = F.unfold(disp * scale, kernel_size=3, padding=1).view(B, 1, 9, 1, 1, H, W)
fine = (mask * patches).sum(dim=2)  # weighted sum over 9-neighbourhood
fine = fine.permute(0, 1, 4, 2, 5, 3).reshape(B, 1, scale*H, scale*W)
```

Note the multiplication `disp * scale` — this is the **disparity unit conversion**. Disparity is measured in pixels at the *current* resolution. Going from 1/4 to 1/2 doubles the number of pixels per row, so the same physical disparity now spans twice as many pixels.

### Why this is the canonical edge-preserving upsample
- It can produce sharp edges (network learns one-sided weighting at depth discontinuities).
- It produces sub-pixel disparities (the convex combination can land between integer values).
- It's cheap (one small mask network plus one matmul-style fold).
- It's differentiable (softmax + linear combination — gradients flow nicely).

This is why both RAFT and RAFT-Stereo use it, and why our StereoLite's `ConvexUpsample` module ([model.py](../../../model/designs/StereoLite/model.py)) is a direct port of the RAFT design.

---

## Training Loss

$$\mathcal{L} = \sum_{i=1}^{N} \gamma^{N-i} \Vert \mathbf{d}_{gt} - \mathbf{d}_i \Vert_1, \quad \gamma = 0.9 \quad \text{(2)}$$

| Symbol | Meaning |
|---|---|
| $\mathcal{L}$ | Total training loss. |
| $N$ | Total iterations during training (22). |
| $i$ | Iteration index (1 to $N$). |
| $\mathbf{d}_i$ | Disparity prediction at iteration $i$. |
| $\mathbf{d}_{gt}$ | Ground-truth disparity. |
| $\Vert \cdot \Vert_1$ | L1 norm — sum of absolute differences. Robust to outliers. |
| $\gamma = 0.9$ | Discount factor. The final iteration has weight $\gamma^0 = 1$; earlier iterations decay. |

This is the **sequence loss** — every iteration is supervised, with later iterations weighted more. Why this matters: gradients flow back through *all* GRU updates, not just the last one, so the network learns to make each iteration useful (not just the final one). Without sequence loss, the early iterations get no learning signal and the network degenerates to using only the last few iterations.

---

## Experimental Results

![Figure 4: Zero-shot generalisation results on ETH3D. RAFT-Stereo is robust to textureless surfaces and overexposure — direct benefits of the iterative paradigm.](../../../figures/RAFTStereo_fig4_qualitative.png)

### Zero-shot generalisation (trained on Scene Flow only)

![Table 1: Synthetic-to-real generalisation. All methods trained on Scene Flow only, tested on KITTI-15, Middlebury (full / half / quarter), and ETH3D.](../../../figures/RAFTStereo_table1_generalization.png)

| Method | KITTI-15 | Middlebury full | Middlebury half | Middlebury quarter | ETH3D |
|--------|---------|----------------|-----------------|-------------------|-------|
| HD³ | 26.5 | 50.3 | 37.9 | 20.3 | 54.2 |
| GwcNet | 22.7 | 47.1 | 34.2 | 18.1 | 30.1 |
| PSMNet | 16.3 | 39.5 | 25.1 | 14.2 | 23.8 |
| GANet | 11.7 | 32.2 | 20.3 | 11.2 | 14.1 |
| DSMNet | 6.5 | 21.8 | 13.8 | 8.1 | 6.2 |
| **RAFT-Stereo** | **5.74** | **18.33** | **12.59** | **9.36** | **3.28** |

RAFT-Stereo wins 4 of 5 settings and is the **first method to process Middlebury at full resolution** — previous methods OOM.

### KITTI-2015
At publication time, **2nd** on the KITTI-2015 leaderboard among published methods — first non-3D-cost-volume method to crack the top.

### Middlebury
**#1** at submission — first architecture since HSMNet (2019) to handle full-res Middlebury, by a wide margin.

---

## Why RAFT-Stereo Changed Everything

Five reasons the field pivoted on this paper:

1. **No expensive 3D convolutions.** The correlation volume is 1D and computed with one matmul. All operations are 2D.
2. **No fixed disparity range.** Iterative refinement reaches any disparity by stepping toward it, no hard cap baked in.
3. **Strong generalisation.** The light architecture plus correlation-only matching transfers across datasets much better than 3D-cost-volume methods.
4. **Handles high resolution.** First architecture to process full-resolution Middlebury and megapixel images.
5. **Flexible speed-accuracy trade-off.** Early stopping works perfectly: 4 iterations gives reasonable results, 32 iterations gives SOTA. You pick.

Virtually every stereo paper from 2022 onward uses RAFT-Stereo as its baseline. DEFOM-Stereo, FoundationStereo, MonSter, Stereo Anywhere, Selective-Stereo — all build directly on the RAFT-Stereo skeleton, replacing or augmenting specific components while keeping the iterative GRU update core intact.

---

## What StereoLite borrows from RAFT-Stereo (and what it deliberately doesn't)

StereoLite is *not* a RAFT-Stereo variant. It's a HITNet-paradigm tile-hypothesis network with two RAFT borrowings. Verified against `model/designs/StereoLite/model.py`.

| RAFT-Stereo component | StereoLite (current) | Reason |
|---|---|---|
| Feature encoder, 256 ch | GhostConv 24/48/72/96 ch (edge), YOLO26s 32/128/256/256 ch (mid) | Edge param budget. RAFT's 256-ch encoder alone is bigger than our entire chassis. |
| Context encoder (separate CNN, left only) | **None.** Matching features are reused as context. | Saves ~0.3 M params. Listed as a future improvement worth ~2-3 days of effort. |
| All-pairs 1D correlation pyramid | **No.** Group-wise cost volume at 1/16, max_disp = 24. Single level, no pyramid. | All-pairs at 1/4 would dominate memory. Group-wise + coarse-resolution + small disparity range fits the budget. |
| 4-level pyramid lookup with linear interp | **No.** Cost volume is consumed by 3D aggregator + parabolic-fit soft-argmin in TileInit. | Different consumption pattern (one-shot init, not iterative lookup). |
| Convolutional GRU update operator | **Stateless 3-layer 2D conv** (`tile_propagate.py:TileRefine`). No hidden state across iterations. | Smaller and faster, but the lack of memory is a known weakness — listed as the largest open improvement. |
| 22-32 GRU iterations | 8 total tile-refinement passes (2 at 1/16, 3 at 1/8, 3 at 1/4). | Each pass at coarser resolution is much cheaper than RAFT's per-iteration cost. |
| Sequence loss with γ=0.9 | **No.** Multi-scale L1 (one per scale), not per-iteration. | Adding this is on the open-improvements list — pure training-side change, no extra params. |
| Multi-level coupled GRUs | **No.** Each scale has its own independent TileRefine. | Tile-state propagation across scales replaces the cross-resolution information flow. |
| Convex upsample (RAFT-style 9-neighbour mask) | **Yes** — `ConvexUpsample` in `model.py`, applied 1/4 → 1/2 → full. | Direct port. The single most useful piece of RAFT machinery for an edge model. |
| Plane-equation upsample | **Yes** (HITNet-derived, not RAFT) — `TileUpsample` between 1/16 → 1/8 → 1/4. | Disparity slopes (sx, sy) propagate naturally across scales without rebuilding cost volumes. |

In summary: StereoLite borrows **convex upsample at the final 2× steps** and **iterative residual updates as a concept** from RAFT, and combines them with **HITNet's tile-state machine and plane-equation upsample**. We're not a RAFT variant.

---

## Connections to Other Papers

| Paper | Relationship |
|---|---|
| **RAFT (ECCV 2020)** | Direct ancestor for optical flow. RAFT-Stereo specialises it to stereo. |
| **CREStereo (2022)** | First major extension — adaptive group correlation + cascaded recurrent refinement. |
| **IGEV-Stereo (2023)** | Combines RAFT's iterative updates with a pre-built geometry encoding volume (hybrid). |
| **Selective-Stereo (2024)** | Replaces the GRU with a Selective Recurrent Unit that fuses multi-frequency branches. Most directly attacks the smearing problem. |
| **IGEV++ (2025)** | IGEV with multi-range volumes and adaptive patch matching. |
| **DEFOM-Stereo (2025)** | Adds foundation-model features and a Scale Update on top of RAFT-Stereo base. |
| **FoundationStereo (2025)** | Combines RAFT-Stereo iteration with a Disparity Transformer for cost filtering. |
| **MonSter (2025)** | Uses IGEV backbone plus bidirectional refinement with monocular depth. |
| **Fast-FoundationStereo (2026)** | Compresses FoundationStereo (and by extension RAFT-Stereo) for real-time. |
