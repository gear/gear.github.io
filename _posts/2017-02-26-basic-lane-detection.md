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
pixel having the coordiate `(x=0,y=0)`. Following this system, the
bottom right corner `img[-1,-1]` has the coordinate 
`(x=img.shape[1]-1, y=img.shape[0]-1)` in OpenCV.

![OpenCV Image]({{site.baseurl}}/img/image_coo.png){:width="50%"}

OpenCV provides various tools for us to "get our hand dirty"
with images. Generally, there are two main groups: image drawing and
image transformation. [Drawing on images in OpenCV](http://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html) is quite simple
and straight forward (except for the elipse :smile:) as most drawing 
function is in the form:

```python
import cv2
cv2.(img_to_draw_on, starting_point, **others_aguments)
```


## Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
