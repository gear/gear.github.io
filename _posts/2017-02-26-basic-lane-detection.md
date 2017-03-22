---
layout: post
type: mini-project
title: Finding Lane Lines on the Road
subtitle: Getting started with OpenCV
---

---
## OpenCV Toolbox

OpenCV is an image processing toolbox originally developed in C++.
In Python, an OpenCV image is a `numpy` array (2D or 3D depending
on the type of image). The figure below depicts the coordination
used in OpenCV. For example, if we have a numpy array `img` describing
an OpenCV image, then `img[0,0]` stores the data for top left
pixel having the coordinate `(x=0,y=0)`. Following this system, the
bottom right corner `img[-1,-1]` has the coordinate
`(x=img.shape[1]-1, y=img.shape[0]-1)` in OpenCV.

![OpenCV Image]({{site.baseurl}}/img/image_coo.png){:width="50%" .center-small}

OpenCV provides various tools for us to "get our hand dirty"
with images. Generally, there are two main groups: image drawing and
image transformation. [Drawing on images in OpenCV](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html) is quite simple
and straight forward (except for the ellipse :smile:) as most drawing
function is in the form:

```python
import cv2
cv2.(img_to_draw_on, starting_point, **others_arguments)
```

## Line Detection Pipeline

The main tool to detect lines in an image is a technique named
_Hough Transformation_. It is called _transformation_ because
it transforms the representation of a line to a pair of line angle
and line distance to the origin. The detail and tutorial of
Hough Transformation is provided
[here](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html). To reduce the computation and increase
the accuracy of the line detection pipeline, we focus only on the
area in front of a car. The following list describes our pipeline:

1. **Extract the region of interest (ROI)**. In this case, the ROI is a trapezoid in front of the car.
2. **Create a color mask**. Lucky for us, the lane lines of interest only have white or yellow colors. Therefore, extracting only yellow and white color will greatly reduce the computation for unwanted objects in the image.
3. **Canny edge detection**. Performing the Canny edge detection algorithm on the color filtered image greatly reduces number of points needed to process in the next step.
4. **(Probabilistic) Hough line transform**. Given a set of points from edge detection, we detect lines using Hough transformation.
5. **Split lines into left and right set**. The output of Hough line transformation is a set of lines, represented by two points `(x1,y1)` and `(x2,y2)`. For each pair of points in the returned set, we split them into left lane points and right lane points by its coefficient.
6. **Fitting lines to left and right points**. We get two lines for left lane and right lane by fitting a line to each set of points. The output of OpenCV's line fitting algorithm is a 4-tuple: `(x0, y0, vx, vy)`, where `(x0,y0)` is a point on the line and `(vx,xy)` is the line's co-linear vector.
7. **Drawing lines onto the image**. To draw a line onto an image, we need two points (start and end). We can compute these points for drawing from the output of step 6 and a specified drawing zone (the trapezoid ROI for example).

---

## Reflection

### Color space

The first pipeline we came up with in this project simply converts the
input to gray scale, detect edges by Canny algorithm, and then draw all
line that Hough transformation returns. This approach is unstable since
it depends heavily on the high and low threshold of the Canny algorithm.
Although using a slight Gaussian blur on the gray scale image can
reduce noise and improve the quality of the pipeline, imperfection on
the road can potentially disturb the pipeline's robustness.

The unused information in our first pipeline is: 1. The region of
interest, and 2. The color of the lane lines. By extracting the region
of interest, we eliminate unnecessary computation.

![OpenCV Image]({{site.baseurl}}/img/focus_region.png){:width="50%" .center-small}

###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ...

Another shortcoming could be ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
