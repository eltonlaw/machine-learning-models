Optical Character Recognition using Expert Systems
---

Algorithm breakdown:

1. Load image
2. Identify symbols in image (Optional - I was using sprites)
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

---------------------------------

To get a better sense of how things work, here are the first 10 symbols as they go through the system:

**1)** Load image

**2)** Symbols seperated into individual images (Optional - I was using sprites)

![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/0_symbols/0.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/0_symbols/1.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/0_symbols/2.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/0_symbols/3.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/0_symbols/4.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/0_symbols/5.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/0_symbols/6.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/0_symbols/7.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/0_symbols/8.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/0_symbols/9.jpg?raw=true)

**3-1)** Horizontal Convolution using a (5, 1) vector filled with 0.2 as the mask

![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/1_convolution1/0.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/1_convolution1/1.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/1_convolution1/2.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/1_convolution1/3.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/1_convolution1/4.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/1_convolution1/5.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/1_convolution1/6.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/1_convolution1/7.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/1_convolution1/8.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/1_convolution1/9.jpg?raw=true)

**3-2)** Vertical Convolution using (1, 5) column vector filled with 0.2 as the mask

![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/2_convolution2/0.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/2_convolution2/1.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/2_convolution2/2.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/2_convolution2/3.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/2_convolution2/4.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/2_convolution2/5.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/2_convolution2/6.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/2_convolution2/7.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/2_convolution2/8.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/2_convolution2/9.jpg?raw=true)

**3-3)** Binarize image using Otsu Thresholding and Scale to 32x32

![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/3_scaled/0.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/3_scaled/1.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/3_scaled/2.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/3_scaled/3.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/3_scaled/4.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/3_scaled/5.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/3_scaled/6.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/3_scaled/7.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/3_scaled/8.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/3_scaled/9.jpg?raw=true)

**3-4)** Zhang-Suen Thinning 

![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/4_thinned/0.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/4_thinned/1.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/4_thinned/2.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/4_thinned/3.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/4_thinned/4.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/4_thinned/5.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/4_thinned/6.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/4_thinned/7.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/4_thinned/8.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/4_thinned/9.jpg?raw=true)

**3-5)** Canny Edge Detection

![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/5_edges/0.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/5_edges/1.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/5_edges/2.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/5_edges/3.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/5_edges/4.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/5_edges/5.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/5_edges/6.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/5_edges/7.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/5_edges/8.jpg?raw=true)
![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/5_edges/9.jpg?raw=true)

**3-6)** 3 x 3 Feature Vector

_Only displaying the first one (cause the image is pretty big), check [here](https://github.com/eltonlaw/ml_algorithms/tree/master/ocr/outputs/6_feature_vectors) to see the rest_

![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/6_feature_vectors/0.jpg?raw=true)



### Training

At this point it might be good to check out the actual [source code](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/ocr.py)

Training is really simple. Our feature descriptions are a placed into a python dictionary, one key for each label. Each key is initialized with an empty list.

	self.features = {'0':[], '1':[], '2':[], '3':[] ... '9':[]}

Each 3x3 feature vector in our training set is added to it's corresponding label in the `self.features` dictionary. 
	
	fd = FeatureDescriptions()
	for feature,label in feature_label_pairs[:split]:
        fd.add(label,feature)   

Note: The `split` variable is the index that we split the train images from the test images.

### Prediction

To classify new images, the system feeds a list of test images into our trained `FeatureDescriptions` object and calculates the Euclidean distance to every stored feature vector.  

    for feature,label in feature_label_pairs[split:]:
        distances = fd.predict(feature)
        prediction = np.argmin(distances)

The label with the feature vector closest to the new image is selected as the prediction. With 35 training images and 15 test images, this system achieves a 37.5% accuracy rate.

![](https://github.com/eltonlaw/ml_algorithms/blob/master/ocr/outputs/7_predictions.png?raw=true)

### Closing Remarks

Experts systems are an old technique and we can see that they perform poorly on handwritten digits. From a computational perspective, calculating pairwise euclidean distances is a nightmare. Additonally, this system does not account for translational invariance. As an example, if you look at (3-6), the 2 was not centered which could cause problems, say if we had another 2 but it was right justifed. However, I'd like to mention that this system managed to squeeze out 37.5% accuracy despite only 35 training images, which is less feasible with a statistical system. 

---

### References 

Ahmed, M., & Ward, R. K. (2000). An expert system for general symbol recognition. Pattern Recognition, 33(12), 1975-1988. doi:10.1016/s0031-3203(99)00191-0
