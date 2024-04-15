''' Face Data Gathering to train model '''

import cv2
import os

face_name = input('\n Enter your name : ')
face_id = input('\n Enter ID : ')

# Define the base directory path
base_directory = "Dataset"

# Create the directory path for the dataset using .join()
Image_folder = os.path.join(base_directory, f"{face_name}_{face_id}")

# Create the directory if it doesn't exist
os.makedirs(Image_folder, exist_ok=True)

print("\n Initializing face capture. Look at the camera and wait.... ")
cam = cv2.VideoCapture(0)
cam.set(3, 640)     # set video width
cam.set(4, 480)     # set video height

face_detector = cv2.CascadeClassifier('Cascade_Classifier/haarcascade_frontalface_default.xml')

count = 0    # Initialize individual sampling face count
while True:
    ret, img = cam.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        crop_img = gray_img[y:y+h, x:x+w]

        # Save the captured image into the dataset directory
        cv2.imwrite(os.path.join(Image_folder, f"{face_name}_face_{count}.jpg"), crop_img)
        
        cv2.imshow('image', img)
        count += 1

    k = cv2.waitKey(500) & 0xff    # it will capture 2 images per second
    if k == 27:
        break           # Press 'Esc' to exit
    elif count > 30:    # Take 30 face samples and stop video
        break

print("\n Successfully Completed.... ")
cam.release()
cv2.destroyAllWindows()
