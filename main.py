from preprocess import preprocess_data
from model import build_model, detect_potholes
import matplotlib.pyplot as plt
import cv2

train_data_dir = "./dataset/train/"
test_data_dir = "./dataset/test/"

# load and preprocess training/testing data
print('Preprocessing...')
X_train, y_train = preprocess_data(train_data_dir)
X_test, y_test = preprocess_data(test_data_dir)

# check the shapes of the datasets
print('Done!')
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

model = build_model()

# train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# input and output image paths
input_image_path = input('Image to predict: ')
output_image_path = input('Output Name: ')

# detect potholes and create the output image
detect_potholes(input_image_path, output_image_path, model)

# display the output image
output_image = cv2.imread(output_image_path)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()