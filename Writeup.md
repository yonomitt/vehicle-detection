**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[cars]: ./images/cars.png "Example car images from the training set"
[noncars]: ./images/not_cars.png "Example non-car images from the training set"
[cars_hog]: ./images/cars_hog.png "Example HOG of car images from the training set"
[noncars_hog]: ./images/not_cars_hog.png "Example HOG of non-car images from the training set"
[potential0]: ./images/potential0.png "Potential matches detected"
[potential1]: ./images/potential1.png "Potential matches detected"
[potential2]: ./images/potential2.png "Potential matches detected"
[potential3]: ./images/potential3.png "Potential matches detected"
[potential4]: ./images/potential4.png "Potential matches detected"
[potential5]: ./images/potential5.png "Potential matches detected"
[detect_stages0]: ./images/detect_stages0.jpg "Example image going through the pipeline"
[detect_stages1]: ./images/detect_stages1.jpg "Example image going through the pipeline"
[detect_stages2]: ./images/detect_stages2.jpg "Example image going through the pipeline"
[detect_stages3]: ./images/detect_stages3.jpg "Example image going through the pipeline"
[detect_stages4]: ./images/detect_stages4.jpg "Example image going through the pipeline"
[detect_stages5]: ./images/detect_stages5.jpg "Example image going through the pipeline"
[video]: ./project_output.mp4

---

I have provided an [HTML](./Vehicle-Detection.html) file with all of the cells run, for convenience.

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 10th code cell of the IPython notebook. This is found under the section titled *1. Feature selection*.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][cars]
![alt text][noncars]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][cars_hog]
![alt text][noncars_hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and always seemed to come back to the initial set I tried based on the lessons.

 - Orientations: 9
 - Pixels per cell: 8 x 8
 - Cells per block: 2 x 2

It could be that with more test data and time, I might have found a more optimal set of parameters. These, however, worked quite well.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training the classifier is in the 16th code block cell, which is under the section labeled `3. Train a classifier`.

The features for the training data are extracted in the 12th code block cell and the training and test data is split in the 15th.

I trained a linear SVM using features extracted from spacial binning, color histogram, and HOG. I experimented just using some of these, but found that the accuracy of the classifier was always highest when all three were used in conjunction.

The final feature set used for training the classifier included 6,108 features:

| Technique     | Features   | 
|:-------------|-------------:| 
| Spacial binning | 768        | 
| Color histogram | 48      |
| HOG  | 5,292 |
| Total  | 6,108  |

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My sliding window search can be found in the function `potential_cars` in the 25th code block cell, which is found under the section *1. Search for matches*.

I used the fairly basic sliding search window described in the lesson, where the HOG of the image is calculated just once and the processed on a window basis like the original image.

I experimented with scales for the sliding window and found a good accuracy to performance tradeoff with the values [1.0, 1.5].

I restricted the search in the image to `y` values between [400, 656].

I could have potentially made this search more efficient by searching smaller scales toward the top of the image and larger scales at the bottom. 

Additionally, I thought about restricting the search to areas around previously detected cars and the edges of the images where cars could enter the scene. However two things spoke against this:

1. This may not be ideal as if for some reason data from the camera is interrupted, you wouldn't want to miss cars that "suddenly" appear where the algorithm is no long searching
2. Time

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I played around with several different image formats including `YCrCb`, `YUV`, and `LUV`. I tried between 2 and 3 different scales for the sliding windows and experimented with various parameters.

While I wish I could have spent more time optimizing the speed of the classifier, that became a secondary goal to accuracy that I did not have time to fully explore.

![alt text][potential0]
![alt text][potential1]
![alt text][potential2]
![alt text][potential3]
![alt text][potential4]
![alt text][potential5]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heat map and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heat map.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

In an attempt to smooth out the bounding boxes and remove even more false positives, I created a system of combining previous frames' heat maps with the current one and thresholding on the combined heat map. The thinking behind this is that over several frames, the combined heat map would stabilize more and give a better average.

I attempted to tweak how many previous frames to consider and what the new threshold should be. This helped to some extent, but not as much as I was hoping.

### Here are six frames and their corresponding heat maps:

![alt text][detect_stages0]
![alt text][detect_stages1]
![alt text][detect_stages2]
![alt text][detect_stages3]
![alt text][detect_stages4]
![alt text][detect_stages5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline still shows the occasional false positive. This is usually at the edges of the images. Limiting the search window sizes to the appropriate height in the image may help with this.

Additionally a more robust temporal consideration of detected cars would help with this. One thing I would really like to do is average sizes of bounding boxes between frames and discard anything that significantly grows or shrinks beyond normal amounts.

During the [Traffic Sign Recognizer project](https://github.com/yonomitt/traffic-sign-classifier), I had a lot of success using a [Contrast Limited Adaptive Histogram Equalization](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html). I thought this would also be a good situation to use it. I attempted using the CLAHE as the input to the HOG, but found it to give worse results than just using the `YCrCb` or `YUV` images.

It might be a good idea to explore using the CLAHE in another fashion with the classifier as some additional features.

I would have also liked to try this algorithm out using my convolutional network from the Traffic Sign Recognizer project, but ran out of time.

So many fun experiments to try. So little time! C'est la vie!
