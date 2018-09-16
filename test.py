#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 02:43:04 2018

@author: andywang
"""

import os 
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import synthesize_text
import pygame
from pygame import mixer
mixer.init()

def text2speech(text):
    print(text)
    synthesize_text.synthesize_text(text)
    mixer.music.load("/Users/andywang/htn-pointy-thing/output.mp3")
    mixer.music.play()
    while mixer.music.get_busy():   
        pygame.time.Clock().tick(5)
    print("1")
text2speech("Computer lab is fun")

text2speech("A")