#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 02:43:04 2018

@author: andywang
"""


import subprocess
text2speechpath = "synthesize_text.py"
text2speech = text2speechpath  + "--text 'hello'"
subprocess.Popen(text2speech, shell=True)
