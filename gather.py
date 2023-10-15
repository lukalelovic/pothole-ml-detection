from geopy.geocoders import Nominatim
from PIL import Image
import matplotlib.pyplot as plt
import requests
import random
import io

api_key = ''
geolocator = Nominatim(user_agent="csi4352project")

min_latitude = 40.5  # minimum latitude (New York City)
max_latitude = 40.9  # maximum latitude (New York City)
min_longitude = -74.05  # minimum longitude (New York City)
max_longitude = -73.85  # maximum longitude (New York City)
save = ''

addresses = []
i = 3

# generate random coordinates
while len(addresses) < 327:
  latitude = random.uniform(min_latitude, max_latitude)
  longitude = random.uniform(min_longitude, max_longitude)
  
  # perform reverse geocoding to get the address
  location = geolocator.reverse(f"{latitude}, {longitude}")

  # no location found? skip
  if (location is None):
    continue
  
  address = location.address
  # check if the address is not already in the list (to avoid duplicates)
  if address not in addresses:
    addresses.append(address)

# generate streetview image of each latitude, longitude
for address in addresses:
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
  print(response.status_code)
  if (response.status_code >= 300):
    continue

  # open the image using Pillow
  image = Image.open(io.BytesIO(response.content))

  # display the image using matplotlib
  plt.imshow(image)
  plt.axis('off')  # turn off axis labels
  plt.show()

  save = input('save image?')
  if (save == 'y'):
    print('Saving image...')

    # save the image to a file
    with open('./road-dataset/sidewalk/'+str(i)+'.jpg', 'wb') as f:
      f.write(response.content)
    response.close()

    print('Saved!')
    i += 1
    save = ''