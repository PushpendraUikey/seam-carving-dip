# Seam Carving

Content-aware image resizing (Avidan & Shamir, 2007), implemented from scratch in Python/NumPy for the CS-663 Digital Image Processing course project at IIT Bombay.

A seam is an 8-connected path of pixels, one per row (or column). Removing the lowest-energy seam shrinks the image by one pixel while leaving salient content alone. The minimum seam is found with dynamic programming:

$$M(i,j) = e(i,j) + \min\big(M(i-1,j-1),\ M(i-1,j),\ M(i-1,j+1)\big)$$

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
python scripts/carve_demo.py data/samples/synthetic.png -n 80 -o results/out.png
```

## Layout

```
seamcarve/
  energy.py   backward (gradient) energy, own Sobel with replicated borders
  dp.py       cumulative-energy DP: naive ground truth + row-vectorised version, backtracking
  carve.py    seam removal, reduce_width / reduce_height; insertion (phase 3)
  masks.py    protect / remove masks, object removal (phase 3)
  order.py    optimal seam ordering via the transport map (phase 4)
scripts/      command-line demos
tests/        pytest suite (DP checked against brute force and against the naive DP)
data/         datasets (git-ignored except samples), see data/README.md
results/      generated outputs (git-ignored)
```

OpenCV is used only for image I/O, colour conversion and as a reference in tests. Energy, DP, carving, masks and ordering are written from scratch.

## Status and roadmap

Deadline: 9 Nov 2026.

| Dates | Phase | Status |
|---|---|---|
| 26-27 Oct | Setup, backward energy, naive DP baseline | done |
| 28-30 Oct | Numba DP, true forward-energy DP (gate: identical seams to naive DP on 100 images) | |
| 31 Oct - 1 Nov | Seam insertion, protect/remove masks (gate: no protected pixel removed) | |
| 2 Nov | Transport-map seam ordering, CLI, feature freeze | |
| 3-4 Nov | Evaluation on RetargetMe, ablations, runtime comparison | |
| 5-6 Nov | Extension: multi-size images via index maps | |
| 7-9 Nov | Slides, report, submission | |

## References

1. S. Avidan, A. Shamir. Seam carving for content-aware image resizing. *ACM TOG* 26(3), Art. 10, 2007 (SIGGRAPH 2007). doi:[10.1145/1276377.1276390](https://doi.org/10.1145/1276377.1276390)
2. M. Rubinstein, A. Shamir, S. Avidan. Improved seam carving for video retargeting. *ACM TOG* 27(3), 2008 (SIGGRAPH 2008). doi:[10.1145/1360612.1360615](https://doi.org/10.1145/1360612.1360615)
3. M. Rubinstein, D. Gutierrez, O. Sorkine, A. Shamir. A comparative study of image retargeting. *ACM TOG* 29(6), Art. 160, 2010 (SIGGRAPH Asia 2010). Benchmark: [RetargetMe](https://people.csail.mit.edu/mrub/retargetme/)
4. A. Shamir, S. Avidan. Seam carving for media retargeting. *Communications of the ACM* 52(1):77-85, 2009. doi:[10.1145/1435417.1435437](https://doi.org/10.1145/1435417.1435437)
5. M. Rubinstein, A. Shamir, S. Avidan. Multi-operator media retargeting. *ACM TOG* 28(3), 2009. doi:[10.1145/1531326.1531329](https://doi.org/10.1145/1531326.1531329)
