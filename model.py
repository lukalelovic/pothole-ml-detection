# model code
import tensorflow as tf
from preprocess import normalize
from keras import layers, models
import numpy as np
import cv2

def double_conv_block(x, n_filters):
  x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
  x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
  return x

def downsample_block(x, n_filters):
  # Conv2D twice with ReLU activation
  f = double_conv_block(x, n_filters)

  p = layers.MaxPool2D(2)(f)
  p = layers.Dropout(0.3)(p)
  return f, p

def upsample_block(x, conv_features, n_filters):
  # upsample
  x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
  # concatenate
  x = layers.concatenate([x, conv_features])
  # dropout
  x = layers.Dropout(0.3)(x)
  # Conv2D twice with ReLU activation
  x = double_conv_block(x, n_filters)
  return x

# define U-Net model
def build_model(size=(128, 128, 3)):
  inputs = layers.Input(shape=size)

  # encoder: contracting path - downsample
  f1, p1 = downsample_block(inputs, 64)
  f2, p2 = downsample_block(p1, 128)
  f3, p3 = downsample_block(p2, 256)
  f4, p4 = downsample_block(p3, 512)

  # bottleneck
  bottleneck = double_conv_block(p4, 1024)

  # decoder: expanding path - upsample
  u6 = upsample_block(bottleneck, f4, 512)
  u7 = upsample_block(u6, f3, 256)
  u8 = upsample_block(u7, f2, 128)
  u9 = upsample_block(u8, f1, 64)

  # outputs
  outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)

  # unet model with Keras Functional API
  unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

  return unet_model

# function to perform pothole detection and create an output image
def detect_potholes(model, image_path):
  # load and preprocess the input image
  img = cv2.imread(image_path)
  img = cv2.resize(img, (128, 128))
  img = img / 255.0

  # create output image with segmentation highlighting
  prediction = model.predict(img)
  return prediction.squeeze()  # remove batch dimension