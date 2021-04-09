import app_utils
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


class_names = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']


IMAGE_SIZE = 512
# define the test configuration
class TestConfig(Config):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = 1 + 46
	NAME = "fashion"
	BACKBONE = 'resnet50'
	IMAGE_MIN_DIM = IMAGE_SIZE
	IMAGE_MAX_DIM = IMAGE_SIZE    
	IMAGE_RESIZE_MODE = 'none'
	RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    
rcnn = MaskRCNN(mode='inference', model_dir='.', config=TestConfig())
rcnn.load_weights('mask_rcnn_fashion_0003.h5', by_name=True)



def main():
	option = st.selectbox(
	'What are you looking for ?',
	('shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'hat', 'headband, head covering, hair accessory', 'shoe', 'bag, wallet'))
	st.write('You selected:', option)

	cloth_id = {'shirt, blouse':0, 'top, t-shirt, sweatshirt':1, 'sweater':2, 'cardigan':3, 'jacket':4, 'vest':5, 'pants':6, 'shorts':7, 'skirt':8, 'coat':9, 'dress':10, 'hat':14, 'headband, head covering, hair accessory':15, 'shoe':23, 'bag, wallet':24}
	search_cloth = cloth_id[option]
	st.title("Fashion AI")
	st.subheader("Your clothes research vision assistant")
	image_file = st.sidebar.file_uploader("Upload Image",type=['png','jpeg','jpg'])
	if image_file is not None:
		st.image(image_file,width=250,height=250)
		image = np.array(Image.open(image_file))
		img = app_utils.white_balance(image)
		img = app_utils.sharpen_img(img)
		img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
		img2 = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
		results = rcnn.detect([img], verbose=0)
		r = results[0]
		i = 0
		class_id = list(r['class_ids'])
		while i < len(class_id):
			class_id[i] = class_id[i]-1
			i += 1
		if search_cloth in class_id:
			cloth_to_search = class_id.index(search_cloth)
		else:
			pass
		r['class_ids'] = np.array(class_id)
		mask = r['masks']
		mask = mask.astype(int)
		img[:,:,2] = img[:,:,2] * mask[:,:,cloth_to_search]
		image1 = img[:,:,2][r['rois'][cloth_to_search][0]:r['rois'][cloth_to_search][2], r['rois'][cloth_to_search][1]:r['rois'][cloth_to_search][3]]
		image = img2[r['rois'][cloth_to_search][0]:r['rois'][cloth_to_search][2], r['rois'][cloth_to_search][1]:r['rois'][cloth_to_search][3]]
		blur = cv2.blur(image1,(5,5))
		th, im_th = cv2.threshold(blur, -0, 256, cv2.THRESH_BINARY)
		im_th = cv2.resize(im_th, im_th.shape[1::-1])
		dst = cv2.bitwise_and(image, image, mask=im_th)
		st.image(dst,width=250,height=250)

if __name__ == '__main__':
	main()