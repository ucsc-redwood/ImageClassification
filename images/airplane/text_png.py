from PIL import Image
import numpy as np

def text_to_png(text_file, output_path, shape=(32, 32, 3)):
    with open(text_file, 'r') as file:
        text_data = file.read()
    
    # Convert text data back to numpy array
    flat_image = np.array(list(map(int, text_data.split())))
    image = flat_image.reshape(shape)
    
    # Create image from numpy array
    image = Image.fromarray(image.astype(np.uint8))
    
    # Save image
    image.save(output_path)

# Example usage
text_file = 'image_1.txt'
output_path = 'image_1.png'
text_to_png(text_file, output_path)

