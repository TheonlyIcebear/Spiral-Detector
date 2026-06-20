# Golden-Spiral-Detector
Algorithmically detecting whether the Golden Spiral is actually present in an image not just eyeballed on top of it.

# Information ℹ
This started as a research project (full math derivation is in `paper.pdf` if you want to see the work) after getting annoyed at how often the Golden Spiral gets loosely overlaid on art and nature photos as "proof" the Golden Ratio shows up everywhere. I wanted to know if you could actually measure that instead of just squinting at a spiral overlay until it looks close enough.

The short version: edge-detect an image, map the edge points into the complex plane, and apply a complex logarithm. Logarithmic spirals which the Golden Spiral is one specific case of become straight lines under that transform. So "is this a Golden Spiral" turns into "how close is this line's slope to the value the Golden Ratio predicts."

### Why I made this:
- I kept seeing the same vague "the Golden Spiral fits this seashell/galaxy/painting" claims and wanted a way to actually check them instead of taking it on faith
- Most existing spiral detectors (Hough-based, mostly built for galaxy astronomy) are accurate but basically black boxes to me I wanted something I could derive and fully explain myself, start to finish

<img width="961" height="423" alt="image" src="https://github.com/user-attachments/assets/49e0d01e-e45f-4fff-92b6-36bc7d68a3d5" />



## Development 🔨
- My first idea was nonlinear regression directly on the spiral equation, with rotation, scale, and position as free parameters. That falls apart fast the equation isn't linear in any of those, so the regression gets messy and unstable
- Linearizing through a complex logarithm fixes that, but introduces a new problem: the transform only produces a straight line if you center it on the spiral's actual origin first, and I had no ground-truth center to start from
- I used PCA anisotropy (the ratio of spread along the two principal axes) as a stand-in for "how linear does this look if I center it here" sweep a bunch of candidate centers, keep the one that maximizes anisotropy
 - This mostly worked, but kept throwing a false second peak near the edges of the image. Turns out the complex log's branch cut was slicing a single spiral arm into two disconnected clusters that *looked* linear to my anisotropy metric purely because each cluster was narrow, not because they actually fell on a line together

<img width="539" height="473" alt="image" src="https://github.com/user-attachments/assets/25507edb-bdcf-4c29-b467-fc428902f524" />


 - Fixed it by folding the Bimodality Coefficient into the metric it specifically penalizes split-cluster patterns like that, and downweighting bimodal results pulled the peak back to the true center
- I deliberately avoided RANSAC and Hough Transforms even though they'd probably give cleaner results on noisy images, because I wanted every step of the pipeline to be something I could explain end-to-end instead of a tuned black box doing the heavy lifting for me

## ✨ Features
- **Edge-Based Feature Extraction:** Gaussian blur + Canny edge detection pulls spiral-relevant points out of an image while discarding flat, low-contrast regions that don't carry any shape information
- **Complex-Plane Linearization:** Converts the polar spiral equation (notoriously non-injective and awkward to regress on directly) into a straight line via a complex logarithm, turning spiral-fitting into ordinary linear regression
- **Automatic Spiral Centering:** Sweeps candidate centers and scores each one with a custom linearity metric PCA anisotropy combined with a bimodality penalty to locate the true center with no manual input required
- **Golden Ratio Benchmarking:** Compares the recovered growth rate against the exact value required for a true Golden Spiral, so the output is a number you can check, not just a visual impression
- **Spiral Reconstruction & Overlay:** Projects the best-fit line back into image space and draws it over the original image as a visual sanity check

<img width="786" height="523" alt="image" src="https://github.com/user-attachments/assets/c526caed-cc66-4b92-bc42-535c3ee4fff0" />


---

# Notes
- This is a research/exploration project, not a production-grade CV tool the Canny edge detector's `sigma` still has to be chosen somewhat by hand per image
- Full mathematical derivation, including why the complex logarithm linearizes the spiral and how the centering metric was built, is in [`paper.pdf`](./paper.pdf)
- Tested against a handful of commonly-cited "Golden Spiral in nature" images most came close to the Golden Ratio but not exactly on it, which honestly ended up being the most interesting result of the whole project

# Sources 🔌
- Portman, Matthew, and Wayne B. Hayes. *Using Sparcfire to Automate Galfit's Multi-Component Decomposition of Spiral Galaxies*, 2024. https://doi.org/10.2139/ssrn.5020734
