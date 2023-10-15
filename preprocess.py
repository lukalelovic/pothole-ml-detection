import numpy as np
import os
import cv2

# function to load and preprocess an image and label
def preprocess_data(path, size=(256, 256)):
	images = []
	labels = []

	for folder in os.listdir(path):
		if folder == "positive":
			label = 1  # Potholes
		elif folder == "negative":
			label = 0  # No potholes
		else:
			continue # go deeper through path

		folder_path = os.path.join(path, folder)

		for filename in os.listdir(folder_path):
			if filename.lower().endswith(".jpg"):
				img_path = os.path.join(folder_path, filename)

				# load the image
				img = cv2.imread(img_path)

				# image needs resizing?
				if img.shape[:2] != size:
					# resize image
					img = cv2.resize(img, size)
					cv2.imwrite(img_path, img) # write re-sized image

				img = img / 255.0  # Normalize pixel values to [0, 1]
				images.append(img)
				labels.append(label)
	return np.array(images), np.array(labels)

# todo: augmentation function?