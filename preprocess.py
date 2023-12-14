# preprocessing code
import numpy as np
import glob
import random
import cv2
import os

def resize(image, mask, zoom_factor=1, size=(128, 128)):
  image = cv2.resize(image, size, fx=zoom_factor, fy=zoom_factor)
  mask = cv2.resize(mask, size, fx=zoom_factor, fy=zoom_factor)
  return image, mask

def flip(image, mask):
  if random.random() > 0.5:
    image = cv2.flip(image, 1)
    mask = cv2.flip(mask, 1)
  return image, mask

def rotate(image, mask):
  # apply rotation randomly (10% of time)
  if random.random() > 0.9:
    rotation_angle = random.uniform(-45, 45)  # degree range

    img_matrix = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), rotation_angle, 1)
    image = cv2.warpAffine(image, img_matrix, (image.shape[1], image.shape[0]))

    mask_matrix = cv2.getRotationMatrix2D((mask.shape[1] // 2, mask.shape[0] // 2), rotation_angle, 1)
    mask = cv2.warpAffine(mask, mask_matrix, (mask.shape[1], mask.shape[0]))
  return image, mask

def preprocess_data(path):
  image_path = os.path.join(path, 'images/*.jpg')
  mask_path = os.path.join(path, 'masks/*.png')
  print(image_path, mask_path)

  image_names = glob.glob(image_path)
  image_names.sort()

  mask_names = glob.glob(mask_path)
  mask_names.sort()

  images = []
  masks = []

  for i in range(len(image_names)):
    img = cv2.imread(image_names[i], 0)
    mask = cv2.imread(mask_names[i], 0)

    zoom_factor = 1

    # apply zoom randomly (10% of time)
    if random.random() > 0.9:
      zoom_factor = 1.5

    img, mask = resize(img, mask, zoom_factor=zoom_factor)
    img, mask = flip(img, mask)
    img, mask = rotate(img, mask)

    # binary mask color (0 or 255)
    mask = (mask >= 75).astype(np.uint8) * 255

    images.append(img)
    masks.append(mask)
  
  images = np.array(images)
  images = np.expand_dims(images, axis=3)

  masks = np.array(masks)
  masks = np.expand_dims(masks, axis=3)

  return images, masks