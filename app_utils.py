import streamlit as st
from PIL import Image 
import cv2
import tensorflow as tf 

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import skimage.io
import numpy as np
from tensorflow import keras
import os
import gc
import sys
import json
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
from tqdm import tqdm

# example of inference with a pre-trained coco model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import numpy as np
import cv2



def white_balance(img):
	try:
		if len(img) > 0 and isinstance(img, (np.ndarray)) == True:
			result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
			avg_a = np.average(result[:, :, 1])
			avg_b = np.average(result[:, :, 2])
			result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
			result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
			result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
			return result
		else:
			return False
	except:
		return False

def sharpen_img(img):
	try:
		if len(img) > 0 and isinstance(img, (np.ndarray)) == True:
			kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
			img = cv2.filter2D(img, -1, kernel)
			return img
		else:
			return False
	except:
		return False

def select_cloth(option):
	try:
		if len(option) > 0 and isinstance(option,str) == True and option in ['top, t-shirt, sweatshirt', 'pants', 'shorts', 'skirt', 'dress', 'hat', 'shoe', 'bag, wallet']:
			st.write('You selected:', option)
			cloth_id = {'shirt, blouse':0, 'top, t-shirt, sweatshirt':1, 'sweater':2, 'cardigan':3, 'jacket':4, 'vest':5, 'pants':6, 'shorts':7, 'skirt':8, 'coat':9, 'dress':10, 'hat':14, 'headband, head covering, hair accessory':15, 'shoe':23, 'bag, wallet':24}
			search_cloth = cloth_id[option]
			return search_cloth
		else:
			return False
	except:
		return False

def img_enhance(image):
	try:
		if len(image) > 0 and isinstance(image, (np.ndarray)) == True:
			img = white_balance(image)
			img = sharpen_img(img)
			img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			return img
		else:
			return False
	except:
		return False

def extract_from_mask(r, img, img2, mask, cloth_to_search):
	try:
		img[:,:,2] = img[:,:,2] * mask[:,:,cloth_to_search]
		image1 = img[:,:,2][r['rois'][cloth_to_search][0]:r['rois'][cloth_to_search][2], r['rois'][cloth_to_search][1]:r['rois'][cloth_to_search][3]]
		blur = cv2.blur(image1,(5,5))
		return blur
	except:
		return False

def get_mask(blur, image):
	try:
		th, im_th = cv2.threshold(blur, -0, 256, cv2.THRESH_BINARY)
		im_th = cv2.resize(im_th, im_th.shape[1::-1])
		dst = cv2.bitwise_and(image, image, mask=im_th)
		return dst
	except:
		return False

def generate_class_id(r):
	try:
		class_id = list(r['class_ids'])
		i = 0
		while i < len(class_id):
			class_id[i] = class_id[i]-1
			i += 1
		return class_id
	except:
		return False


def get_cloth_to_search(search_cloth, class_id):
	try:
		if search_cloth in class_id:
			cloth_to_search = class_id.index(search_cloth)
			return cloth_to_search
		else:
			return False
	except:
		return False

def cloth_img_exctraction(image_file, IMAGE_SIZE, rcnn, search_cloth):
	st.image(image_file,width=250,height=250)
	image = np.array(Image.open(image_file))
	img = img_enhance(image)
	img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
	img2 = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
	results = rcnn.detect([img], verbose=0)
	r = results[0]
	class_id = generate_class_id(r)
	cloth_to_search = get_cloth_to_search(search_cloth, class_id)
	r['class_ids'] = np.array(class_id)
	mask = r['masks']
	mask = mask.astype(int)
	blur = extract_from_mask(r, img, img2, mask, cloth_to_search)
	image = img2[r['rois'][cloth_to_search][0]:r['rois'][cloth_to_search][2], r['rois'][cloth_to_search][1]:r['rois'][cloth_to_search][3]]
	dst = get_mask(blur, image)
	st.image(dst,width=250,height=250)


def main(rcnn, IMAGE_SIZE):
	option = st.selectbox('What are you looking for ?',('top, t-shirt, sweatshirt', 'pants', 'shorts', 'skirt', 'dress', 'hat', 'shoe', 'bag, wallet'))
	search_cloth = select_cloth(option)
	st.title("Fashion AI")
	st.subheader("Your clothes research vision assistant")
	image_file = st.sidebar.file_uploader("Upload Image",type=['png','jpeg','jpg'])

	if image_file is not None:
		dst = cloth_img_exctraction(image_file, IMAGE_SIZE, rcnn, search_cloth)
