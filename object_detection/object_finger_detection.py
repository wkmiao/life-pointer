#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 15:18:58 2018

@author: andywang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 01:16:59 2018

@author: andywang
"""


# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:

#from finger detection 
#from __future__ import print_function
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt



from distutils.version import StrictVersion
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image



import myo as libmyo; libmyo.init("/Users/andywang/htn-pointy-thing/sdk/myo.framework/myo")


class Listener(libmyo.DeviceListener):
  def on_connected(self, event):
    print("Hello, '{}'! Double tap to exit.".format(event.device_name))
    event.device.vibrate(libmyo.VibrationType.short)
    event.device.request_battery_level()

  def on_battery_level(self, event):
    print("Your battery level is:", event.battery_level)

  def on_pose(self, event):
    if event.pose == libmyo.Pose.wave_in:
      print ("left")
      return True
    elif event.pose == libmyo.Pose.wave_out:
      print ("right")
      return True
  
    elif event.pose == libmyo.Pose.double_tap:
      return False


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


# ## Env setup

# In[ ]:


# This is needed to display the images.
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Object detection imports
# Here are the imports from the object detection module.

# In[ ]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[ ]:


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[ ]:

'''
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

'''
# ## Load a (frozen) Tensorflow model into memory.

# In[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#initilize for finger detection 
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points

#logictech webcam
greenLower = (45, 54, 63)
greenUpper = (80, 255, 255)

#mac webcam
#greenLower = (29, 86, 6)
#greenUpper = (64, 255, 255)


#pts = deque(maxlen=args["buffer"])

#Creating a Pandas DataFrame To Store Data Point
Data_Features = ['x', 'y', 'time']
Data_Points = pd.DataFrame(data = None, columns = Data_Features , dtype = float)


#Reading the time in the begining of the video.
start = time.time()





# In[ ]:

#intializing the web camera device

#import cv2
cap = cv2.VideoCapture(1)
print("Start")

# Running the tensorflow session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   hub = libmyo.Hub()
   myo_listener = Listener()
   ret = True
   x=0
   while (ret):
      #grab the current frame - finger detection 
      #(grabbed, frame) = camera.read() 
      
      (ret,image_np) = cap.read()
      #code for finger detection
      #Reading The Current Time
      current_time = time.time() - start

      # resize the frame, blur it, and convert it to the HSV
      # color space
      frame = imutils.resize(image_np, width=1800)
      # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
      hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
      # construct a mask for the color "green", then perform
      # a series of dilations and erosions to remove any small
      # blobs left in the mask
      mask = cv2.inRange(hsv, greenLower, greenUpper)
      mask = cv2.erode(mask, None, iterations=2)
      mask = cv2.dilate(mask, None, iterations=2)
      # find contours in the mask and initialize the current
      # (x, y) center of the ball
      cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		 cv2.CHAIN_APPROX_SIMPLE)[-2]
      center = None

      
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
#      plt.figure(figsize=IMAGE_SIZE)
#      plt.imshow(image_np)
#####
      
      
      # only proceed if at least one contour was found
      if len(cnts) > 0:
          # find the largest contour in the mask, then use
          # it to compute the minimum enclosing circle and
          # centroid
          c = max(cnts, key=cv2.contourArea)
          ((x, y), radius) = cv2.minEnclosingCircle(c)
          M = cv2.moments(c)
          center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
          # only proceed if the radius meets a minimum size
          if (radius < 300) & (radius > 10 ) : 
              # draw the circle and centroid on the frame,
              # then update the list of tracked points
              cv2.circle(image_np , (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
              cv2.circle(image_np , center, 5, (0, 0, 255), -1)
              #Save The Data Points
              Data_Points.loc[Data_Points.size/3] = [x , y, current_time]
              # update the points queue
              #pts.appendleft(center)
          if cv2.waitKey(25) & 0xFF == ord('q'):
              x = 1
              #print ([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])
              #print([category_index.get(i) for i in classes[0]])
              #print(type(boxes))
              #print(type(classes))  
              #width, height = image_np.size
          if x < 20 and x > 0: 
              print("Width:")
              frameWidth = cap.get(3)
              print(cap.get(3))
              print("height:")
              frameHeight = cap.get(4)
              print(cap.get(4))
              print(center)
              for index,value in enumerate(classes[0]) :
                  if scores[0,index] > 0.5:
                      print ([category_index.get(value)['name'],boxes[0,index]])
                      #print(scores)
              x=x+1
          else:  
              x = 0   
      hub.run_once(myo_listener.on_event, 50)
        
        
      
      cv2.imshow('image',image_np)
      #cv2.imshow('image',cv2.resize(image_np,(1280,960)))
      #if cv2.waitKey(25) & 0xFF == ord('q'):
          #cv2.destroyAllWindows()
          #cap.release()
          #break

