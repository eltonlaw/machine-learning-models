# Optical Character Recognition using Expert Systems
---

Here's the quick breakdown of what happens:

1. Load image
2. Identify symbols in image
3. For each identified symbol image...
  1. Image Convolution (Horizontal 1D Kernel)
  2. Image Convolution (Vertical 1D Kernel) 
  3. Binarize image using Otsu Thresholding
  4. Scale image to 32x32
  5. Apply Canny Edge Detection
  6. Apply Zhang-Suen Thinning Algorithm
  7. Split symbol into 3x3 matrix and store
4. Split 3x3 feature matrixes into two groups, training and test
5. Store descriptions of training set
6. Model predicts for test set

---
# References 
- https://scholar.google.ca/citations?view_op=view_citation&hl=en&user=qsJmtkMAAAAJ&citation_for_view=qsJmtkMAAAAJ:9yKSN-GCB0IC
