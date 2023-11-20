# preprocessing code
import numpy as np
import glob
import cv2

def readImages(path, isMask=False):
  image_names = glob.glob(path)
  image_names.sort()
  #print(image_names)

  images = []
  for image_name in image_names:
    img = cv2.imread(image_name, 0)
    img = cv2.resize(img, (128, 128))

    if isMask:
      img = (img >= 75).astype(np.uint8) * 255

    images.append(img)

  dataset = np.array(images)
  dataset = np.expand_dims(dataset, axis = 3)
  
  return dataset