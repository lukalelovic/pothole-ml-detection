import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import cv2

# define the CNN model
def build_model(img_width=256, img_height=256):
  model = keras.Sequential([
  	layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
  	layers.MaxPooling2D((2, 2)),
  	layers.Conv2D(64, (3, 3), activation='relu'),
  	layers.MaxPooling2D((2, 2)),
  	layers.Flatten(),
  	layers.Dense(128, activation='relu'),
  	layers.Dense(1, activation='sigmoid')  # binary classification
  ])

  # compile the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

# function to perform pothole detection and create an output image
def detect_potholes(input_image_path, output_image_path, model, img_width=256, img_height=256):
  # load and preprocess the input image
  input_image = cv2.imread(input_image_path)

  # create an output image with segmentation highlighting
  output_image = np.copy(input_image) 
  input_image = cv2.resize(input_image, (img_width, img_height))
  input_image = input_image / 255.0  # normalize pixel values to [0, 1] 

  # perform pothole detection
  prediction = model.predict(np.expand_dims(input_image, axis=0))
  print(prediction)

  color = (0, 0, 255) # red color

  # find contours in prediction
  contours, _ = cv2.findContours(np.uint8(prediction), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # draw rectangles around detected objects
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2) # draw a red rectangle
		
  # save the output image
  cv2.imwrite(output_image_path, output_image)