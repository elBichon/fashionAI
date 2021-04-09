import app_utils
import config_files
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


CLASS_NAMES = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']
IMAGE_SIZE = 512
rcnn = MaskRCNN(mode='inference', model_dir='.', config=config_files.TestConfig())
rcnn.load_weights('mask_rcnn_fashion_0003.h5', by_name=True)


def main():
	option = st.selectbox('What are you looking for ?',('top, t-shirt, sweatshirt', 'pants', 'shorts', 'skirt', 'dress', 'hat', 'shoe', 'bag, wallet'))
	search_cloth = app_utils.select_cloth(option)
	st.title("Fashion AI")
	st.subheader("Your clothes research vision assistant")
	image_file = st.sidebar.file_uploader("Upload Image",type=['png','jpeg','jpg'])

	if image_file is not None:
		st.image(image_file,width=250,height=250)
		image = np.array(Image.open(image_file))
		img = app_utils.img_enhance(image)
		img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
		img2 = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  
		results = rcnn.detect([img], verbose=0)
		r = results[0]
		

		class_id = list(r['class_ids'])
		i = 0
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
		blur = app_utils.extract_from_mask(r, img, img2, mask, cloth_to_search)
		image = img2[r['rois'][cloth_to_search][0]:r['rois'][cloth_to_search][2], r['rois'][cloth_to_search][1]:r['rois'][cloth_to_search][3]]
		dst = app_utils.get_mask(blur, image)
		st.image(dst,width=250,height=250)

if __name__ == '__main__':
	main()