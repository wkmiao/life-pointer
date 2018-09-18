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
import math 
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#sys.path.insert(0, '/Users/andywang/htn-pointy-thing/synthesize_text')
#import synthesize_text 

from pygame import mixer 
import myo as libmyo; libmyo.init("/Users/andywang/htn-pointy-thing/sdk/myo.framework/myo")
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import synthesize_text
import stt
import pygame



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
## Load a (frozen) Tensorflow model into memory.

#text to speech 
mixer.init()
def text2speech(text):
    synthesize_text.synthesize_text(text)
    mixer.music.load("/Users/andywang/htn-pointy-thing/object_detection/output.mp3")
    mixer.music.play()
    while mixer.music.get_busy():   
        pygame.time.Clock().tick(5)

text2speech("Hack the North is legit")


DETECT_OBJECT_FLAG = False
LOOK_FOR_OBJECT_FLAG = False

class Listener(libmyo.DeviceListener):
  def on_connected(self, event):
    print("Hello, '{}'! Double tap to exit.".format(event.device_name))
    event.device.vibrate(libmyo.VibrationType.short)
    event.device.request_battery_level()

  def on_battery_level(self, event):
    print("Your battery level is:", event.battery_level)


  def on_pose(self, event):
    if event.pose == libmyo.Pose.wave_in:
      global leftObject 
      print("Object on the left: " , end=" ")  
      for i in leftObject :
          print(i, end=" ")
      total_text = ". ".join(str(item) for item in leftObject) 
      print(total_text)
      text2speech(total_text)
      return True
    elif event.pose == libmyo.Pose.wave_out:
      global rightObject  
      print("Object on the right: " , end=" ")   
      for i in rightObject :
          print(i, end=" ")
      total_text = ". ".join(str(item) for item in rightObject) 
      print(total_text)
      text2speech(total_text)
      return True
    elif event.pose == libmyo.Pose.fingers_spread:
      # where is ___ (start microphone)
      print("finger spread")
      global LOOK_FOR_OBJECT_FLAG 
      LOOK_FOR_OBJECT_FLAG = True
    elif event.pose == libmyo.Pose.double_tap:
      # what-is-this equivalent 
      print("double tap")
      global DETECT_OBJECT_FLAG
      DETECT_OBJECT_FLAG = True
    elif event.pose == libmyo.Pose.fist:
      #closed fist to stop
      return False

def computePointedObject(objects, fingerPos):
    '''to be checked outside:
    if whatisthis and fingerPos:
    '''
    centres = {}
    box_nums = []
    diags = {}
    limit = len(objects)

    unique_i = 0

    # obj = {'key/label', 'val'=[[center1, diag1], [center2, diag2], []... ]}
    # centres = {'key/label', 'val'=[[unique_center1], [unique_center2], []... ]}

    #iterate through keys/labels
    
    print (objects)
    
    for key, objs in objects.items():
        for o in objs:
            centre = o[0]
            diag = o[1]
        
            if centres == None:
                centres[key] = []
                centres[key].append(centre)
                diags[key] = []
                diags[key].append(diag)
                box_nums = 1
                unique_i = 0
                prev_i = 0
                
                
            else:
                    for i, c in centres.items():
                        # if centre within 10% of c val in centres == duplicate el
                        # else greater than 10% == new el
                        if abs(centre - c) < (0.1 * c):
                            #get sum of centres, diagonals
                            centres[key][i] += centre
                            diags[key][i] += diag
                            box_nums[i] += 1
                        elif unique_i < limit:
                            unique_i += 1
                            box_nums[unique_i] = 1
                            centres.append(centre)
                            diags.append(diag)
                        else:
                            break

    #compute average centres
    for key, boxes in centres.items():
        for i in range(len(boxes)):
            #tuple
            centres[key][i] /= box_nums[i]
            diags[key][i] /= box_nums[i]


    #search for closest
    x = fingerPos[0]
    y = fingerPos[1]


    for key, c in centres.items():
        diag = c[1]
        if diag < 100:
            diag *= 2
        cx =  c[0][0]
        cy =  c[0][1]
        #if inside, return
        if math.sqrt((cx - x)^2 + (cy - y)^2 ) < diag:
            return c.key

    return None




#math.hypot(p2[0] - p1[0], p2[1] - p1[1]) # Linear distance 
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


#starting text to speech 
#text2speechpath = "/Users/andywang/htn-pointy-thing/synthesize_text.py"
#text2speech = text2speechpath  + "--text 'hello'"
#subprocess.Popen(text2speech, shell=True)

mixer.init()
# mixer.music.load("/Users/andywang/htn-pointy-thing/output.mp3")
# mixer.music.play()

# In[ ]:
counter = 0
#intializing the web camera device

#import cv2
cap = cv2.VideoCapture(1)
frameWidth = cap.get(3)
frameHeight = cap.get(4)
print("Start")
print("frameWidth:" )
print(frameWidth)
print("frameHeight:")
print(frameHeight)

save_to_array = False
detected_objects = {}
saved_fingerPos = (0,0)
# Running the tensorflow session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   hub = libmyo.Hub()
   myo_listener = Listener()
   ret = True

   detected_objects = {}
   fingerPos = 0

   while (ret):
      #grab the current frame - finger detection 
      #(grabbed, frame) = camera.read() 
      
      (ret,image_np) = cap.read()
      #code for finger detection
      #Reading The Current Time
      current_time = time.time() - start

      # resize the frame, blur it, and convert it to the HSV
      # color space
      
      #not sure we need it
      #image_np= imutils.resize(image_np, width=1800)
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
          # it to compute the minimum enclosing circle and
          # centroid
          c = max(cnts, key=cv2.contourArea)
          ((x, y), radius) = cv2.minEnclosingCircle(c)
          M = cv2.moments(c)
          center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
          fingerPos = center
          # only proceed if the radius meets a minimum size
          if (radius < 300) & (radius > 10 ): 
              # draw the circle and centroid on the frame,
              # then update the list of tracked points
              cv2.circle(image_np , (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
              cv2.circle(image_np , center, 5, (0, 0, 255), -1)
              #Save The Data Points
              Data_Points.loc[Data_Points.size/3] = [x , y, current_time]
              # update the points queue
              #pts.appendleft(center)
          if DETECT_OBJECT_FLAG and counter == 0:
              counter = 1
              save_to_array = True
              detected_objects = {}
              saved_fingerPos =center
              #print ([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])
              #print([category_index.get(i) for i in classes[0]])
              #print(type(boxes))
              #print(type(classes))  
              #width, height = image_np.size
      leftObject = []
      rightObject = [] 
      # format: [ymin, xmin, ymax, xmax]
      #calculate center of the boxes 

      for index,value in enumerate(classes[0]) :
          if scores[0,index] > 0.5:     
              xmid = (boxes[0,index][1]+boxes[0,index][3])/2
              ymid = (boxes[0,index][0]+boxes[0,index][2])/2
              
              if (xmid < 0.5):
                  leftObject.append(category_index.get(value)['name'])
              else:
                  rightObject.append(category_index.get(value)['name'])
              ymin = boxes[0,index][0]* frameHeight
              xmin = boxes[0,index][1]* frameWidth
              ymax = boxes[0,index][2]* frameHeight
              xmax = boxes[0,index][3]* frameWidth
              xmid = (xmin + xmax)/2
              ymid = (ymin + ymax)/2
              if save_to_array:
                  diag = math.sqrt(pow((ymax  - ymin ),2)  + 
                      pow((xmax - xmin),2) ) / 2

                  if (category_index.get(value)['name']) not in detected_objects:
                      detected_objects[(category_index.get(value)['name'])] = []
                  detected_objects[(category_index.get(value)['name'])].append([(xmid , ymid ), diag])
                  
                #print ([category_index.get(value)['name'],boxes[0,index]])

#      print("Object on the left: " , end=" ")  
#      for i in leftObject :
#          print(i, end=" ")
#      print("Object on the right: "  end=" ")   
#      for i in rightObject :
#          print(i, end=" ")
#     for index,value in enumerate(classes[0]) :
#              if scores[0,index] > 0.5:
#                  print ([category_index.get(value)['name'],boxes[0,index]])
        
      if counter > 0 and counter < 11:
          counter = counter + 1
          DETECT_OBJECT_FLAG = False 
      else:  
          counter = 0  
          DETECT_OBJECT_FLAG = False 
          save_to_array = False
          
      if counter == 10 :
          print("detect object" )
          print(detected_objects)
          print(fingerPos)
          computePointedObject(detected_objects, saved_fingerPos)
          detected_objects = {}
          saved_fingerPos= (0,0)
          
      
      if counter ==0:   
          hub.run(myo_listener.on_event, 50)
        
        
      cv2.imshow('image',image_np)
      #cv2.imshow('image',mask)

      #cv2.imshow('image',cv2.resize(image_np,(1280,960)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release()
          break

