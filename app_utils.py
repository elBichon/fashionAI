import numpy as np
import cv2
import streamlit as st


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