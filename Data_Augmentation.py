import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Path to your dataset
dataset_path = "Dataset/Priyanshu_1"
output_path = "New_Images"

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load and resize images
images = []
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = load_img(os.path.join(dataset_path, filename), target_size=(150, 150))  # Resize images to a common size
        images.append(img)

# Convert images to numpy arrays
images_array = np.array([img_to_array(img) for img in images])

# Create ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate augmented images
num_augmented_images = 0
for img_array in images_array:
    img_array = img_array.reshape((1,) + img_array.shape)
    for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_path, save_prefix='aug', save_format='jpg'):
        num_augmented_images += 1
        if num_augmented_images >= 120:  # Stop after generating 120 augmented images per original image
            break
    if num_augmented_images >= 120:
        break

print("Total augmented images generated:", num_augmented_images)
