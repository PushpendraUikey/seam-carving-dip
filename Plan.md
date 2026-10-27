# Seam Carving Project Plan

Oct 26, 2026 · @Pushpendra Uikey

## Overview

The project builds a from-scratch seam carving library in Python, written to O(nm) per seam and benchmarked against the li-plus reference, then extends it with optimal seam ordering and one research-style extension. The application angle is object and photo-bomber removal, compared against classical inpainting, with ablations showing where seams fail. Everything is derived from DIP course material: gradient operators, image as a 2D signal, and dynamic programming over a pixel graph.

Deliverables:

- A Python package (`seamcarve/`) with backward and forward energy, numba-accelerated DP, seam removal and insertion, protect and remove masks, and transport-map ordering.
- An evaluation notebook with qualitative grids, runtime tables and comparisons against crop and bicubic scaling on the RetargetMe images.
- A 10-minute presentation, with code and results uploaded to Moodle by 9 Nov 2026, plus a short demo (CLI and notebook slider).

Ground rules: write every core function yourself; use OpenCV only for I/O, colour conversion and the Sobel/Scharr reference; cite any repo you read. The gulshan-123 IITB repo stays reference-only because it is an unattributed copy of andrewdcampbell/seam-carving from the same course (663) and copying it is a plagiarism risk.

## Mathematical foundations

Seam carving is a shortest-path problem on an 8-connected pixel graph: the cheapest monotone path under an energy map, found exactly by dynamic programming in O(nm) for an n × m image.

**Energy (backward).** Avidan and Shamir's baseline is the L1 gradient magnitude, computed on luminance with replicate borders (not wrap, which leaks energy across image edges):

```latex
e_1(I) = \left|\frac{\partial I}{\partial x}\right| + \left|\frac{\partial I}{\partial y}\right|
```

Variants to compare: Sobel vs Scharr vs central differences, the L2 magnitude, local entropy (9 × 9 window) and the HoG-normalised e\_HoG = e\_1 / max HoG(I(x, y)) over an 11 × 11 window, all listed in the 2007 paper.

**Seam.** A vertical seam is one pixel per row with a connectivity constraint:

```latex
\mathbf{s}^x = \{(i, x(i))\}_{i=1}^{n}, \quad |x(i) - x(i-1)| \le 1, \qquad s^* = \arg\min_{\mathbf{s}} \sum_{i=1}^{n} e(I(s_i))
```

**DP recurrence (backward energy).** The cumulative cost map is filled row by row; backtracking from the minimum in the last row recovers s\*:

```latex
M(i,j) = e(i,j) + \min\big(M(i-1,j-1),\; M(i-1,j),\; M(i-1,j+1)\big)
```

Each row depends only on the previous one, so a row is a single vectorised min over three shifted arrays (pad with +inf at the borders). Store the argmin offset in a backtrack array of int8 to keep memory at n × m bytes.

**Forward energy (Rubinstein et al. 2008).** Backward energy ignores the new edges created when neighbours become adjacent after removal. Forward energy charges exactly those new edges, using three transition costs:

```latex
\begin{aligned}
C_U(i,j) &= |I(i,j+1) - I(i,j-1)| \\
C_L(i,j) &= C_U(i,j) + |I(i-1,j) - I(i,j-1)| \\
C_R(i,j) &= C_U(i,j) + |I(i-1,j) - I(i,j+1)|
\end{aligned}
```

```latex
M(i,j) = P(i,j) + \min\big(M(i-1,j-1) + C_L,\; M(i-1,j) + C_U,\; M(i-1,j+1) + C_R\big)
```

P(i, j) is an optional pixel prior (mask penalties). The transition cost must sit inside the min, which is the bug in the gulshan-123 code: it precomputes one cost per pixel and then runs a plain backward DP.

**Insertion.** To enlarge by k columns, find the k seams that removal would take (on a copy), then duplicate them in removal order, mapping each to original coordinates and averaging each new pixel with its left and right neighbours. Cap k at about 50% of the width per pass to avoid stretching the same region.

**Object removal.** Set P = −B (a large constant such as −1e5 times the energy scale) on the remove mask and +B on the protect mask, carve in the direction with the smaller mask extent until no mask pixel remains, then insert the same number of seams to restore size.

**Optimal ordering (transport map).** To reach (n − r) × (m − c), the order of horizontal and vertical removals matters. Let T(r, c) be the minimal cost to remove r rows and c columns:

```latex
T(r,c) = \min\Big(T(r-1,c) + E\big(\mathbf{s}^{y}(I_{n-r+1 \times m-c})\big),\; T(r,c-1) + E\big(\mathbf{s}^{x}(I_{n-r \times m-c+1})\big)\Big)
```

with T(0, 0) = 0, where s^y is a horizontal seam and s^x a vertical one. A one-bit choice map backtracks the optimal order. The naive cost is O(rc · nm), so keep only the previous row of images in memory and evaluate it on downscaled inputs.

**Complexity.** One seam costs O(nm) for energy + DP + removal, so k seams cost O(k · nm). A local energy update (recompute only a 2-pixel band around the removed seam) cuts the energy part per seam to O(n); the DP stays O(nm).

## Implementation milestones

Each phase ends with a test that must pass before the next starts, so a bug in the DP never hides behind a later feature.

| Phase | Build | Done when |
| --- | --- | --- |
| 0. Setup | Repo layout (`seamcarve/energy.py`, `dp.py`, `carve.py`, `masks.py`, `order.py`), pytest, 6 to 10 test images, a naive pure-Python DP as ground truth | Naive DP finds the known seam on a hand-built 5 × 5 energy grid |
| 1. Backward energy | e\_1 with Sobel, Scharr and central differences on luminance, replicate borders; L2 and entropy variants | Matches `cv2.Sobel` to 1e-6; constant image gives zero energy everywhere |
| 2. DP and removal | Vectorised row-wise DP (NumPy shifted mins, int8 backtrack), then a `@numba.njit` version; boolean-mask removal for colour images; local energy update | Both versions return the same seam as the naive DP on 100 random images; numba version under 10 ms per seam on 500 × 333 |
| 3. Forward energy | True transition-cost DP from Rubinstein 2008 inside the min | Same seam as a brute-force path search on 6 × 6 random images; visibly fewer jagged edges on the bench image than backward |
| 4. Insertion | k-seam batch insertion with index mapping and neighbour averaging; staged passes for k over 50% of width | Removing then inserting k seams restores the original size; inserted seams never overlap |
| 5. Masks | Protect and remove masks as the prior P; object removal then size restore; horizontal seams by transposing | Remove mask fully gone; no protected pixel ever removed (assert on every seam) |
| 6. Optimal ordering | Transport map T(r, c) with a choice map and backtrack; row-by-row memory | On small images the total cost is at most that of rows-first and cols-first orders |
| 7. CLI and demo | `python -m seamcarve in.jpg --width W --height H --energy forward --mask m.png`; notebook slider | Runs end to end on all test images |

Two engineering details matter for speed. Remove a seam with a boolean mask and a reshape (`img[~mask].reshape(n, m-1, 3)`), never a per-row Python loop. Keep a running index map of original coordinates through every removal; insertion, object removal and the multi-size extension all reuse it.

## Evaluation

Evaluate on the RetargetMe benchmark: 80 images tagged by attribute (lines/edges, faces/people, texture, foreground objects, geometric structures, symmetry), with published results from 8 retargeting methods to compare against ([Rubinstein et al. 2010](https://people.csail.mit.edu/mrub/papers/retBenchmark.pdf)).

| Experiment | Compare | Measure |
| --- | --- | --- |
| Correctness | Vectorised and numba DP vs naive DP | Seam identity on 100 random images; total seam cost equal to 1e-9 |
| Runtime | Naive, NumPy, numba, numba + local energy update, li-plus | ms per seam and total seconds to remove 200 columns at 3 image sizes (e.g. 300, 600, 1200 px wide) |
| Energy ablation | Backward e\_1, e\_HoG, entropy, forward | Visual grid per attribute class; edge discontinuity count along seams |
| Operator comparison | Seam carving vs crop vs bicubic scaling vs RetargetMe results (SC, MULTIOP, WARP) | EMD and SIFTflow distance to the source, the two metrics that agreed best with users in the 2010 study |
| Ordering | Width-first, height-first, transport-map optimal | Total removed energy and visual result at 25% reduction in both axes |
| Failure analysis | Faces, straight lines, dense texture, 50% reduction | Annotated failure cases with the cause (seams crossing structure, repeated insertion) |

If time allows, add a small paired-comparison user study (10 classmates, 10 image pairs, forward vs backward and seam carving vs scaling) and report the preference rate with a binomial test. Runtime results get one chart (ms per seam vs image width, one line per implementation) once measured.

## Extensions

Pick one primary extension and do it well; multi-size images is the recommended choice because it reuses the index map from Phase 2 and gives a striking real-time demo.

| Extension | What to build | Effort | Source |
| --- | --- | --- | --- |
| Multi-size images (recommended) | Index map V where V(i, j) = the seam number that removes pixel (i, j); any width m′ is a threshold V ≥ m − m′, so resizing becomes an O(nm) lookup; negative indices for enlargement; a slider demo | Medium | Avidan and Shamir 2007, Section 5 |
| Text-line extraction | Seams through low-ink gaps between manuscript lines using a distance-transform or smoothed energy; evaluate on a few scanned pages | Low to medium | devaljain repo; Saabni and El-Sana, language-independent text line extraction |
| Video retargeting | Spatio-temporal seams via graph cut, or per-frame forward energy with a temporal smoothness term as a cheaper approximation | High | Rubinstein et al. 2008 |
| Multi-operator retargeting | Combine seam carving, scaling and cropping; choose the operator sequence by minimising a bidirectional image distance | High | Rubinstein et al. 2009 |
| Topologically constrained epitomes | Attention-driven, variable-size image summary | High | Source not found yet; needs a citation before it goes in the report |

Multi-size caveat from the paper: horizontal and vertical index maps cannot be computed independently because seams in one direction break the other map, so support one direction fully and treat the other as a stretch goal.

## Report and timeline

The deadline is 9 Nov 2026, so the 8 phases run in 2 weeks. The deliverable is a 10-minute deck plus code and results on Moodle. Draft each part below as its phase closes; each maps to one or two slides.

1. Introduction: retargeting, why scaling and cropping fail, contributions (1 page).
2. Background: gradient energy, image graphs, dynamic programming, related work from references 1 to 5 (1.5 pages).
3. Method: energy functions, the seam DP with proof of optimality, forward energy, insertion, masks, transport map (3 pages).
4. Implementation: data layout, vectorisation, numba, local energy update, complexity (1 page).
5. Experiments: the evaluation table above, with figures per attribute class (2.5 pages).
6. Extension: multi-size images (1 page).
7. Limitations and failure cases, then conclusion (1 page).
8. References and an appendix on code provenance stating which repos were read and that none was copied.

&#91;embedded content: 8-week roadmap · 7 stages, 4 gates\]

Nov 3 to 9 is the buffer. If the core slips, cut the extension first, then the transport map, and never the evaluation, because the course grades empirical analysis and ablations explicitly.

## References

Both of your papers check out, but each citation mixes two editions: the venue you name and the DOI you give belong to different records. Cite the journal (TOG) version consistently, as below.

**Core papers (verified against Crossref)**

1. S. Avidan and A. Shamir, "Seam carving for content-aware image resizing," *ACM Trans. Graph.*, vol. 26, no. 3, Art. 10, 2007. doi:10.1145/1276377.1276390. ([author PDF](https://www.eng.tau.ac.il/~avidan/papers/imretFinal.pdf), [project page](https://faculty.runi.ac.il/arik/site/seam-carve.asp)). Your DOI is the TOG record; "SIGGRAPH 2007 papers, 10-es" is the proceedings record, doi:10.1145/1275808.1276390.
2. M. Rubinstein, A. Shamir and S. Avidan, "Improved seam carving for video retargeting," *ACM Trans. Graph.*, vol. 27, no. 3, 2008. doi:10.1145/1360612.1360615. ([author PDF](https://www.eng.tau.ac.il/~avidan/papers/vidret.pdf), [project page](https://faculty.runi.ac.il/arik/site/seam-video.asp)). Your DOI 10.1145/1399504.1360615 is the "SIGGRAPH 2008 papers" record, not TOG 27(3).

**Added trusted references**

3. M. Rubinstein, A. Shamir and S. Avidan, "Multi-operator media retargeting," *ACM Trans. Graph.*, vol. 28, no. 3, 2009. doi:10.1145/1531326.1531329. ([author PDF](https://www.eng.tau.ac.il/~avidan/papers/multiop-lowres.pdf))
4. M. Rubinstein, D. Gutierrez, O. Sorkine and A. Shamir, "A comparative study of image retargeting," *ACM Trans. Graph.* (Proc. SIGGRAPH Asia), vol. 29, no. 6, pp. 160:1 to 160:10, 2010. ([paper](https://people.csail.mit.edu/mrub/papers/retBenchmark.pdf), [RetargetMe dataset](https://people.csail.mit.edu/mrub/retargetme/), [survey page](https://faculty.runi.ac.il/arik/site/ret-survey.asp)). DOI not yet confirmed.
5. A. Shamir and S. Avidan, "Seam carving for media retargeting," *Commun. ACM*, vol. 52, no. 1, pp. 77 to 85, 2009. doi:10.1145/1435417.1435437. A readable overview for the report's background section.
6. A. Das, "Improved seam carving with forward energy," 2019 ([blog](https://avikdas.com/2019/07/29/improved-seam-carving-with-forward-energy.html)). A worked explanation of the forward-energy costs; useful for checking your derivation, not for citing as a primary source.

**Code (read, cite, never copy)**

- [li-plus/seam-carving](https://github.com/li-plus/seam-carving/) ([PyPI](https://pypi.org/project/seam-carving)): MIT licence; backward and forward energy, `keep_mask`, `drop_mask`, width-first or height-first order. Reports about 1 s to remove 200 columns from a 600 × 407 image. Use it as the runtime baseline.
- [devaljain/Text-Line-Extraction-Using-Seam-Carving](https://github.com/devaljain/Text-Line-Extraction-Using-Seam-Carving): exists; Python with NumPy, Matplotlib and Pillow, a single `seamcarving.py`, 4 commits. It cites no paper, so pair it with Saabni and El-Sana's [language-independent text line extraction](https://cris.bgu.ac.il/en/publications/language-independent-text-lines-extraction-using-seam-carving-4/) if you use it.
- gulshan-123/IIT-Bombay-CSE, Sem5/663-DIP/project: reference only. It is an unattributed copy of andrewdcampbell/seam-carving submitted for this same course.

Open item: no trusted source found yet for "topologically constrained epitomes"; find the original paper before citing it.

Sources checked on 2026-10-26: Crossref records for items 1 to 3 and 5, the RetargetMe paper and BibTeX from the [Universidad de Zaragoza graphics group](https://graphics.unizar.es/docs/Rubinstein10Comparative.bib), and the two GitHub repository pages.
