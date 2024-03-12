from PIL import Image

def print_image_shape(image_path):
    image = Image.open(image_path)
    image_shape = image.size
    print("Image shape:", image_shape)

if __name__ == "__main__":
    image_path = 'original_image.png'
    print_image_shape(image_path)

