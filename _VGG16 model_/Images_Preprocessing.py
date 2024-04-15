import cv2
import os

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('Cascade_Classifier/haarcascade_frontalface_default.xml')

# Path to the folder containing images
folder_path = "Dataset/BillGates_9"

# Create a directory to store cropped images if it doesn't exist
cropped_folder = "New_Images"
os.makedirs(cropped_folder, exist_ok=True)

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
        # Read the image
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        
        # Convert the image to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        # Crop and save each detected face
        for i, (x, y, w, h) in enumerate(faces):
            face_img = img[y:y+h, x:x+w]
            # Save the cropped face as grayscale
            cropped_face_path = os.path.join(cropped_folder, f"{folder_path[8:-2]}_face_{i}.jpg")
            cv2.imwrite(cropped_face_path, cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY))
