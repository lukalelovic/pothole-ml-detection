from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import requests
import numpy as np
import io
import cv2
import os

load_dotenv()

choice1 = input('Are you uploading a street view image? (y/n) ')
img_name = ''
image = None

if choice1 == 'y':
  img_name = input('Path to image: ')

  image = cv2.imread(img_name, 0)

  if image is None:
    print('Image not found:', img_name)
else:
  confirm = 'n'
  address = ''

  while confirm == 'n':
    street = input('Enter the street address: ')
    city_state = input('Enter the city, state: ')

    address = street + ', ' + city_state
    confirm = input('Is this correct? (y/n) ', address)

  # load api key from env file
  api_key = os.getenv('API_KEY')

  # street view parameters
  size = '400x400'  # image size
  fov = '90'        # field of view (zoom level)
  heading = '0'     # camera heading (0 is north)
  pitch = '0'       # camera pitch (0 is horizontal)

  print('Getting street view data at '+address+'')

  # construct the Street View image URL
  url = f'https://maps.googleapis.com/maps/api/streetview?location={address}&size={size}&fov={fov}&heading={heading}&pitch={pitch}&key={api_key}'

  # send a GET request to the API
  response = requests.get(url)
  print('Response:', response.status_code)

  # open the image using Pillow
  image = Image.open(io.BytesIO(response.content))

  # display the image using matplotlib
  plt.imshow(image)
  plt.axis('off')  # turn off axis labels
  plt.show()

  save = input('Save image? (y/n) ')
  if (save == 'y'):
    img_name = input('Image name: ')
    print('Saving image...')

    img_name = './'+img_name+'.jpg'

    # save the image to a file
    with open(img_name, 'wb') as f:
      f.write(response.content)
    response.close()

    print('Saved!')
    save = ''
  
  image = cv2.imread(img_name, 0)

image = cv2.resize(image, (128, 128))

# perform preprocessing on image
image = cv2.resize(image, (128, 128))
image = image / 255.

model = load_model('pothole_segmentation_model.h5')

test_img = np.expand_dims(image, 0)
prediction = (model.predict(test_img)[0,:,:,0] > 0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Image')
plt.imshow(image, cmap='gray')
plt.subplot(232)
plt.title('Predicted Mask')
plt.imshow(prediction, cmap='gray')

plt.show()