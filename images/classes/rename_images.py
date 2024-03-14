import os
import shutil

# Function to rename images in a directory and keep them in the same folder
def rename_images_in_folder(folder_path):
    class_names = os.listdir(folder_path)
    for class_name in class_names:
        class_dir = os.path.join(folder_path, class_name)
        if os.path.isdir(class_dir):
            image_files = os.listdir(class_dir)
            for i, image_file in enumerate(image_files):
                old_image_path = os.path.join(class_dir, image_file)
                new_image_name = f"{class_name}_{i + 1}.png"
                new_image_path = os.path.join(class_dir, new_image_name)
                shutil.move(old_image_path, new_image_path)
                print(f"Renamed: {old_image_path} -> {new_image_path}")

# Provide the path to the directory containing the images
folder_path = "/home/riksharm/ImageClassification/images/classes"

# Rename images in each class subdirectory
rename_images_in_folder(folder_path)

print("Images renamed successfully.")

