# model code
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, MaxPool2D, Concatenate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def conv_block(input, num_filters):
  x = Conv2D(num_filters, 3, padding="same")(input)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  
  x = Conv2D(num_filters, 3, padding="same")(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  
  return x

# Encoder block: Conv block followed by maxpooling
def encoder_block(input, num_filters):
  x = conv_block(input, num_filters)
  p = MaxPool2D((2, 2))(x)
  p = Dropout(0.3)(p)

  return x, p   

# Decoder block: skip features gets input from encoder for concatenation
def decoder_block(input, skip_features, num_filters):
  x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
  x = Concatenate()([x, skip_features])
  x = Dropout(0.3)(x)
  x = conv_block(x, num_filters)
  return x

# Build Unet using the blocks
def build_unet(input_shape, n_classes):
  inputs = Input(input_shape)
  
  s1, p1 = encoder_block(inputs, 64)
  s2, p2 = encoder_block(p1, 128)
  s3, p3 = encoder_block(p2, 256)
  s4, p4 = encoder_block(p3, 512)
  
  b1 = conv_block(p4, 1024) # Bottleneck
  
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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def confusionMatrix(model, X_train, y_train):
  # Predictions
  y_train_pred = model.predict(X_train)

  # Threshold predictions to convert probabilities to binary values
  y_train_pred_binary = np.round(y_train_pred)

  # Compute confusion matrix
  conf_matrix_train = confusion_matrix(y_train.flatten(), y_train_pred_binary.flatten())

  # Convert confusion matrix to percentages
  conf_matrix_train_percentage = conf_matrix_train / conf_matrix_train.sum() * 100

  # Compute precision and recall for training set
  precision_train = conf_matrix_train[1, 1] / (conf_matrix_train[1, 1] + conf_matrix_train[0, 1])
  recall_train = conf_matrix_train[1, 1] / (conf_matrix_train[1, 1] + conf_matrix_train[1, 0])
  balanced_accuracy_train = 0.5 * (recall_train + precision_train)

  print('Precision (Training Set):', precision_train)
  print('Recall (Training Set):', recall_train)
  print('Balanced Accuracy (Training Set):', balanced_accuracy_train)

  # Plot confusion matrix for training set
  plt.figure(figsize=(6, 6))
  disp_train = ConfusionMatrixDisplay(conf_matrix_train_percentage, display_labels=[0, 1])
  disp_train.plot(cmap='viridis', values_format='.2f', ax=plt.gca(), colorbar=False)
  plt.title('Confusion Matrix - Training Set')

  # Display balanced accuracy rate in a box on the graph
  plt.figtext(0.5, 0.01, f'Balanced Accuracy (Training Set): {balanced_accuracy_train:.4f}', 
                wrap=True, horizontalalignment='center', fontsize=10)
    
  plt.show()