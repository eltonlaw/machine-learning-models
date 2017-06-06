# k-Nearest Neighbours

Hyperparameters

* `nn`: Number of Neighbours (integer)

Broadly, this is how the algorithm works. For each test set datapoint, `x`...

1. Calculate the distance of `x` to every vector in the train set.
2. Find the labels of the `nn` closest vectors in the train set.
3. Predict the label of `x` by somehow aggregating the results of step 2 (there are different methodologies).  


## References:

https://www.youtube.com/watch?v=hAeos2TocJ8&index=2&list=PLlJy-eBtNFt6EuMxFYRiNRS07MCWN5UIA