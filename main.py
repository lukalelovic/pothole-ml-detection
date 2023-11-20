from model import build_unet, calcLoss
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from preprocess import readImages
import numpy as np
import matplotlib.pyplot as plt
import random
import os

train_data_dir = "./cracks-and-potholes-in-road"
model_path = "./pothole_segmentation_model.h5"

# load and preprocess training/testing data
print('Preprocessing...')

# load images and masks
img_dataset = readImages('./cracks-and-potholes-in-road/images/*.jpg')
mask_dataset = readImages('./cracks-and-potholes-in-road/masks/*.png', True)

print("Image shape:", img_dataset.shape)
print("Mask shape:", mask_dataset.shape)
print("Image max:", np.max(img_dataset))
print("Mask labels:", np.unique(mask_dataset))

img_dataset = img_dataset/255. # normalize
mask_dataset = mask_dataset/255. # rescale

X_train, X_test, y_train, y_test = train_test_split(img_dataset, mask_dataset, test_size = 0.20, random_state = 42)

# sanity check, view few masks
image_number = random.randint(0, len(X_train)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(X_train[image_number,:,:,0], cmap='gray')
plt.subplot(122)
plt.imshow(y_train[image_number,:,:,0], cmap='gray')
plt.show()

if os.path.exists(model_path):
  # load the existing model from the .h5 file
  model = load_model(model_path)
  print("Model loaded from", model_path)
else:
  IMG_HEIGHT = img_dataset.shape[1]
  IMG_WIDTH  = img_dataset.shape[2]
  IMG_CHANNELS = img_dataset.shape[3]
  
  input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

  model = build_unet(input_shape=input_shape, n_classes=1)

  model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

  # get model parameters
  print(model.summary())

  NUM_EPOCHS = 32
  BATCH_SIZE = 32
  STEPS_PER_EPOCH = 25

  # train the model
  hist = model.fit(X_train, y_train, batch_size=BATCH_SIZE, verbose=1, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_data=(X_test, y_test), shuffle=False)

  # save the model
  model.save('pothole_segmentation_model.h5')

  calcLoss(hist)

# IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_pred_thresholded, y_test)
print("Mean IoU =", IOU_keras.result().numpy())

threshold = 0.5
test_img_number = random.randint(0, len(X_test)-1)
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)
print(test_img_input.shape)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
print(prediction.shape)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()