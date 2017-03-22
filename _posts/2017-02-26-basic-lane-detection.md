---
layout: post
type: mini-project
title: Finding Lane Lines on the Road
subtitle: Getting started with OpenCV
---

<i class="fa fa-github"></i> [p1-lanelines on Github](https://github.com/gear/CarND/tree/master/lanelines-p1)

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

### Color space and region of interest

The first pipeline we came up with in this project simply converts the
input to gray scale, detect edges by Canny algorithm, and then draw all
line that Hough transformation returns. This approach is unstable since
it depends heavily on the high and low threshold of the Canny algorithm.
Although using a slight Gaussian blur on the gray scale image can
reduce noise and improve the quality of the pipeline, imperfection on
the road can potentially disturb the pipeline's robustness.

The unused information in our first pipeline is: 1. The region of
interest, and 2. The color of the lane lines. By extracting the region
of interest, we eliminate unnecessary computation:

![Focus region]({{site.baseurl}}/img/focus_region.png){:width="70%" .center-small}

In addition to extracting the region of interest, we also filtered out
unwanted colors. By default, the image output of `cv2.imread` is a GBR
image. This color representation makes it hard to filter a certain
color since all three values (G,B,R) of a pixel represents color.
Therefore, we convert the image to HSV color space
(Hue, Saturation, Value). In HSV images, a pixel contains the color
(hue), the "amount" of that color (saturation), and its brightness
(value). This color representation enable us to specify the colors we
want to extract. To exact colors, the rule of thumb is to range
&plusmn;10 in the hue value as following:

```python
hue_range = 10  # Increase for wider color selection
# rgb_color is the (R,G,B) tuple value of the color we want to filter
pixel = np.uint8([[rgb_color]])  # One pixel image
hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)  # Convert to HSV
hue = hsv_pixel[0,0,0]  # Get the hue value of the input (R,G,B)
lowb = np.array((hue-hue_range, 100, 100), dtype=np.uint8)
upb = np.array((hue+hue_range, 255, 255), dtype=np.uint8)
return lowb, upb  # Lower and upper bound for color filtering
```

To exact black or white color, the code is different since it depends
on the saturation and value rather than the hue.

```python
sensitivity = 30
lowwhite = np.array((0,0,255-sensitivity), dtype=np.uint8)
upwhite = np.array((255,sensitivity,255), dtype=np.uint8)
return lowwhite, upwhite  # Lower and upper bound for color filtering
```

After selecting only the region of interest and the colors, we have
the following result:

![Focus region]({{site.baseurl}}/img/filtered_roi.png){:width="70%" .center-small}

The image above is a binary image which can be used as a mask to
extract the lane lines from the original image. The example of our
lane lines detection on static image is shown below.

![Result on image]({{site.baseurl}}/img/result_lanelines.png){:width="70%" .center-small}

###  Buffered pipeline

The pipeline showed in the previous session performs well on test
images and videos. However, with the challenge video, it failed to
detect the lane lines for some brief moments when the lighting varies.
Furthermore, in all videos, the lane lines between frame doesn't have
smooth transitions. To address this problem, we have several approaches:

1. Limit the movement of lines between frames. We specify a limit
$\alpha$ for the displacement of two lines between adjacent frames. The
next frame's line is computed as: $x_t = x_{t-1} + \min{(\alpha, x_t -
x_{t-1})}$
2. Store previous lines in a fixed-size buffer, add new line to the
buffer for every frame. The output is the weighted average of all the
lines in the buffer.
3. Similar to the second approach, but instead of storing line points,
we store the lines' co-linear vectors. The next line's co-linear vector
is the weighted average of the vectors stored in the buffer.

The videos result for each of the approach will be updated soon. TODO:
Upload videos.

We have some minor bugs during the implementation of of the buffered
pipeline. Firstly, when the buffer is empty, the pipeline should not
draw the line. In one of our implementation, a default line is drawn
when the buffer is empty, this design decision makes it hard to debug
the program. Secondly, when no line is detected from the frame, the
algorithm should still return a line from the buffer. However, if there
are many "no line" frames, there is a chance that there isn't any lane
lines on the road. Our current buffer implementation hasn't taken care
of this situation.

### Unnecessary operations

For the current testing data (images and videos), extracting the region
of interest and lane line colors is enough for line detection.

![Result on image]({{site.baseurl}}/img/only_color.png){:width="70%" .center-small}

As the picture above has shown, only yellow (left) and white (right)
color filter is enough to give us a substantially clear image of lane
lines. This output here can be put directly to the Hough Line detection
(without masking with the original image or Canny edge detection) to
obtain the lane lines. At this stage, we don't know if performing Canny
edge detection is necessary (i.e. makes the pipeline more robust).
