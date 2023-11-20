# model code
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization
from keras.layers import Activation, MaxPool2D, Concatenate
import matplotlib.pyplot as plt

def conv_block(input, num_filters):
  x = Conv2D(num_filters, 3, padding="same")(input)
  x = BatchNormalization()(x)   #Not in the original network. 
  x = Activation("relu")(x)
  
  x = Conv2D(num_filters, 3, padding="same")(x)
  x = BatchNormalization()(x)  #Not in the original network
  x = Activation("relu")(x)
  
  return x

# Encoder block: Conv block followed by maxpooling
def encoder_block(input, num_filters):
  x = conv_block(input, num_filters)
  p = MaxPool2D((2, 2))(x)
  return x, p   

# Decoder block: skip features gets input from encoder for concatenation
def decoder_block(input, skip_features, num_filters):
  x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
  x = Concatenate()([x, skip_features])
  x = conv_block(x, num_filters)
  return x

# Build Unet using the blocks
def build_unet(input_shape, n_classes):
  inputs = Input(input_shape)
  
  s1, p1 = encoder_block(inputs, 64)
  s2, p2 = encoder_block(p1, 128)
  s3, p3 = encoder_block(p2, 256)
  s4, p4 = encoder_block(p3, 512)
  
  b1 = conv_block(p4, 1024) #Bridge
  
  d1 = decoder_block(b1, s4, 512)
  d2 = decoder_block(d1, s3, 256)
  d3 = decoder_block(d2, s2, 128)
  d4 = decoder_block(d3, s1, 64)
  
  if n_classes == 1: # Binary
    activation = 'sigmoid'
  else:
    activation = 'softmax'

  outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)  #Change the activation based on n_classes
  print(activation)

  model = Model(inputs, outputs, name="U-Net")
  return model

def calcLoss(hist):
  # get model loss
  loss = hist.history['loss']
  val_loss = hist.history['val_loss']
  epochs = range(1, len(loss) + 1)
  plt.plot(epochs, loss, 'y', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()

  acc = hist.history['accuracy']
  val_acc = hist.history['val_accuracy']
  plt.plot(epochs, acc, 'y', label='Training acc')
  plt.plot(epochs, val_acc, 'r', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()