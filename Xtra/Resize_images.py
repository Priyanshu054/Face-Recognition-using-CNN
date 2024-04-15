from PIL import Image
import os

def resize_images(folder_path, output_folder, target_size=(240, 240)):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        try:
            # Open the image
            with Image.open(os.path.join(folder_path, filename)) as img:
                # Resize the image
                resized_img = img.resize(target_size)
                # Save the resized image to the output folder
                resized_img.save(os.path.join(output_folder, filename))
                print(f"Resized {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Example usage
folder_path = "Dataset/TonyStark_4"
output_folder = "New_Image"
resize_images(folder_path, output_folder)
