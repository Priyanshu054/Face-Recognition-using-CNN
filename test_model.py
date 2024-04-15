import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# Load the CNN model
cnn_model = load_model("Trained_Model/CNN_Model.h5")

# Load and preprocess the image using OpenCV
img_path = 'Dataset/Priyanshu_1/Priyanshu_face_0.jpg'
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_resized = cv2.resize(img_gray, (120, 120))
img_array = img_to_array(img_gray_resized)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize pixel values to range [0, 1]

# Make prediction
prediction = cnn_model.predict(img_array)
print(prediction)
