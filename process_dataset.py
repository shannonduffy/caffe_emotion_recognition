import os, shutil, sys, time, re, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image #import Image
import caffe

from caffe_functions import *
from opencv_functions import *
from utility_functions import *


dirJaffe = 'datasets/jaffe'
dataset = 'jaffe'
cropFlag = True # False disables image cropping
plot_confusion = True

dir = dirJaffe
color = False
single_face = True
cropFlag = True
useMean = False

categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']

input_list, labels = importDataset(dir, dataset, categories)

classify_emotions(input_list, color, categories, labels, plot_neurons=False, plot_confusion=plot_confusion, useMean=useMean)
