from PIL import Image
import numpy as np

def png_to_text(image_path, output_path):
    image = Image.open(image_path)
    image = np.array(image)
    
    # Print the shape of the original image
    print("Original Image shape:", image.shape)
    
    # Flatten the image
    flat_image = image.flatten()
    
    # Convert the flattened image to text
    text_data = ' '.join(map(str, flat_image))
    
    # Save text data to file
    with open(output_path, 'w') as file:
        file.write(text_data)
    
    # Print the shape of the text data
    print("Text Data shape:", flat_image.shape)

# Example usage
image_path = 'image_1.png'
output_path = 'image_1.txt'
png_to_text(image_path, output_path)

