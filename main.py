from preprocess import preprocess_data
from model import build_model, detect_potholes
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import tensorflow as tf
import cv2

train_data_dir = "./cracks-and-potholes-in-road/"

# load and preprocess training/testing data
print('Preprocessing...')
X_train, y_train = preprocess_data(train_data_dir)

y_train = tf.argmax(y_train, axis=-1)

# check the shapes of the datasets
print('Done!')
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

model = build_model()

model.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# get model parameters
print(model.summary())

# train the model
hist = model.fit(X_train, y_train, epochs=1, batch_size=8, steps_per_epoch=100, verbose=1, validation_split=0.2)

# save the model
model.save('pothole_segmentation_model.h5')

# get model loss
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# input and output image paths
input_image_path = input('Image to predict: ')
output_image_path = input('Output Name: ')

# detect potholes and create the output image
prediction = detect_potholes(model, input_image_path)

# Visualize the original image and the segmentation mask
original_img = cv2.im_read(input_image_path)
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(prediction, cmap='viridis')  # adjust the colormap based on your needs
plt.title('Segmentation')

plt.show()