# receive image and process
# download model
# process image with model

import numpy as np
import os
import tensorflow as tf
import base64

from google.cloud import storage
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from PIL import Image
import base64
from io import BytesIO
from keras.models import load_model

# We keep model as global variable so we don't have to reload it in case of warm invocations
model = None

# Download model file from cloud storage bucket
def download_model_file():

    from google.cloud import storage

    # Model Bucket details
    BUCKET_NAME        = "messy-clean-imgrecog"
    PROJECT_ID         = "face-to-face-fatigue"
    GCS_MODEL_FILE     = "modelku.h5"

    # Initialise a client
    client   = storage.Client(PROJECT_ID)
    
    # Create a bucket object for our bucket
    bucket   = client.get_bucket(BUCKET_NAME)
    
    # Create a blob object from the filepath
    blob     = bucket.blob(GCS_MODEL_FILE)
    
    folder = '/tmp/'
    if not os.path.exists(folder):
      os.makedirs(folder)
    # Download the file to a destination
    blob.download_to_filename(folder + "modelku.h5")


# Main entry point for the cloud function
def bedroom_predict(request):

    # Use the global model variable 
    global model
    if not model:

        download_model_file()
        # model = pickle.load(open("/tmp/local_model.pkl", 'rb'))
        model = load_model(open("/tmp/modelku.h5", 'rb'))
    
    
    # Get the features sent for prediction
    params = request.get_json()
    im = Image.open(BytesIO(base64.b64decode(params)))
    im.save('image1.jpg', 'JPG')

    if (params is not None):
        # Run a test prediction
        # predicting images
        img = image.load_img('image1.jpg')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        images = np.vstack([x])
        classes = model.predict_classes(images, batch_size=10)
        if classes==0:
          category = 'clean'
        else:
          category = 'messy'
        # pred_species  = model.predict(np.array([params['features']]))
        return category
      # return pred_species[0]
        
    else:
        return "nothing sent for prediction"

# def hello_world(request):
#     """Responds to any HTTP request.
#     Args:
#         request (flask.Request): HTTP request object.
#     Returns:
#         The response text or any set of values that can be turned into a
#         Response object using
#         `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
#     """
#     request_json = request.get_json()
#     if request.args and 'message' in request.args:
#         return request.args.get('message')
#     elif request_json and 'message' in request_json:
#         return request_json['message']
#     else:
#         return f'Hello World!'
