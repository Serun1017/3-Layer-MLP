import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt

from MLP import MLP
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 100

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

test_loader = DataLoader(dataset=mnist_test,
                         batch_size=batch_size,
                         shuffle=True)

# Model Load
model = torch.load('model.pth')

# Evaluate Model
model.eval()
correct = 0
total = 0

with torch.no_grad() :
    for images, labels in test_loader :
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images).to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('accuracy:', 100 * correct / total)

# Model Test with Test Image
data_number = 10
with torch.no_grad() :
    sample_images = random.sample(list(mnist_test), data_number)

    for images, labels in sample_images :
        # Print label
        print("Label:", labels)

        # Image visualize
        plt.imshow(images[0], cmap='gray')
        plt.show()

        # Predict Number with Model
        images = images.view(-1, 28*28).to(device)
        labels = labels

        outputs = model(images).to(device)
        _, predicted = torch.max(outputs.data, 1)
        print("Predicted:", predicted.item())
        print("")