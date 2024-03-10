import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def save_image_to_text_and_file(filename, image, class_name):
    # Print the class or type of the image
    print("Image class:", class_name)
    
    with open(filename + '.txt', 'w') as f:
        for c in range(image.size(0)):
            for h in range(image.size(1)):
                for w in range(image.size(2)):
                    # Write each pixel value to the file
                    f.write(f'{image[c][h][w]} ')
        f.write('\n')
    
    # Save the image in PNG format
    transform = transforms.ToPILImage()
    image_pil = transform(image)
    image_pil.save(filename + '.png')

# Transform to convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load the CIFAR-10 training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Extract a single image (change index for different images)
image, label = trainset[0]
class_name = trainset.classes[label]

# Save the image to a text file and PNG file
save_image_to_text_and_file('cifar_image', image, class_name)

