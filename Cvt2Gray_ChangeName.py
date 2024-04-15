import cv2
import os

# Path to the folder containing images
folder_path = "Dataset/ElonMusk_8"

# Create a directory to store cropped images if it doesn't exist
new_folder = "New_Images"
os.makedirs(new_folder, exist_ok=True)

count = 0
# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image

        # Read the image
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        
        # Convert the image to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
        new_face_path = os.path.join(new_folder, f"{folder_path[8:-2]}_face_{count}.jpg")

        # Save the cropped face as grayscale
        # cv2.imwrite(new_face_path, gray)

        cv2.imwrite(new_face_path, img)
        count += 1