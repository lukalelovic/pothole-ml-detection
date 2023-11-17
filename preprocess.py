# preprocessing code
import numpy as np
import random
import os
import cv2

size = (128, 128)

def resize(input_image, image_mask):
	input_image = cv2.resize(input_image, size)
	image_mask = cv2.resize(image_mask, size)
	return input_image, image_mask

def augment(input_image, image_mask):
	if random.random() > 0.5:
		input_image = cv2.flip(input_image, 1)
		image_mask = cv2.flip(image_mask, 1)
	return input_image, image_mask

def normalize(input_image, image_mask):
	input_image = input_image / 255.0
	image_mask -= 1
	return input_image, image_mask

# function to load and preprocess an image and label
def preprocess_data(path):
	images = []
	masks = []

	imgs_path = os.path.join(path, "images")
	masks_path = os.path.join(path, "masks")
 
	for file in os.listdir(imgs_path):
		if not file.lower().endswith(".jpg"):
			continue

		img_path = os.path.join(imgs_path, file)
		mask_path = os.path.join(masks_path, file.replace(".jpg", ".png"))
		
		img = cv2.imread(img_path) # load the image
		mask = cv2.imread(mask_path) # load the mask
	
		# call augmentation functions
		img, mask = resize(img, mask)
		img, mask = augment(img, mask)
		img, mask = normalize(img, mask)
		
		if img is not None:
			images.append(img)
			masks.append(mask)

	return np.array(images), np.array(masks)