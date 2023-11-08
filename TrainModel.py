import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import csv

from MLP import MLP
from torch.utils.data import DataLoader

# Use CUDA if Available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
input_size = 784 # MNIST data image of shape 28 * 28 = 784
hidden_size = 397
output_size = 10 # 0 ~ 9

training_epochs = 30 # training time
batch_size = 100 # data sample number per train
learning_rate = 0.001 

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(), 
                          download=True)


# Dataset Loader
train_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True)


# Model Load
model = MLP(input_size, hidden_size, output_size)
model.to(device)

# Loss Function
criterion = nn.CrossEntropyLoss().to(device)

# Optimizer using Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# FILe to save the epoch and loss data
csv_file = 'epoch_loss_data.csv'

with open(csv_file, 'w', newline='') as file :
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Loss"])
        
    # Training Loop
    for epoch in range(training_epochs):
        avg_loss = 0
        total_batch = len(train_loader)

        for images, labels in train_loader :
            # convert as device
            images = images.view(-1, 28*28).to(device)
            labels = labels.to(device)

            # train with model
            optimizer.zero_grad()
            outputs = model(images).to(device)

            # backpropagation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate average loss
            avg_loss += loss / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.9f}'.format(avg_loss))

        writer.writerow([epoch, avg_loss.item()])


# Test Data Loader
mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

test_loader = DataLoader(dataset=mnist_test,
                         batch_size=batch_size,
                         shuffle=False)

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


# Model Save
torch.save(model, 'model.pth')