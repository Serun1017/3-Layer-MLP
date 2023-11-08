# 3-Layer MLP

---

3-Layer MLP code using pyTorch

세종대학교 소프트웨어학과 19011625 허진수

[![Video Label](https://img.youtube.com/vi/86hd3KZ0VbA/0.jpg)](https://www.youtube.com/watch?v=86hd3KZ0VbA)

---
## MLP Description
### MNIST dataset

MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing int the field of machine learning. It was created by re-mixing the samples from NIST’s original datasets. The black and white images from NIST were normalized to fit into a 28 x 28 pixel bounding box and anti-aliased, which introduced grayscale levels.

The MNIST dataset contains 60,000 training images and 10,000 testing images. The set of images in the MNIST dataset was created in 1994 as a combination of two of NIST’s databases; Special Database 1 and Special Database 3. Special Database 1 and Special Database 3 consist of digits written by high school students and employees of the United States Census Bureau.

---

### MLP

A Multilayer Perception (MLP) is a modern feedforward artificial neural network, consisting of fully connected neurons with a nonlinear kind of activation function, organized in at least three layers, notable for being able to distinguish data that is not linearly separable.
3-Layer MLP consists of three layers; an input layer, a hidden layer and an output layer.

Input layer is the first layer in a neural network. It serves as the interface between the external data and the network itself. The Primary purpose of the input layer is to receive input data and pass it to the subsequent layers for processing.

Hidden Layer is an intermediate layer located between the input layer and the output layer. It plays a crucial role in transforming and learning complex patterns within the input data.

Output Layer is the final layer of neurons that produces the network’s predictions or outputs. It is responsible for providing the model’s response based on the patterns and information learned during training.

---

### Training

During the training phase of the 3-Layer MLP on the MNIST dataset, the model adjusts its weights and biases using the training data to optimize its performance. In this process, the model learns to minimize the loss function by utilizing images and labels from the training data. Throughout the training phase, the model employs the backpropagation algorithm to update its weights and fine-tune its predictions on the data to find the optimal weights.

---

### Testing

After the training is completed, the 3-Layer MLP model evaluates its performance on the test data. During this phase, the model takes the test data images as input and makes predictions, which are then compared to the actual labels. The model’s accuracy is determined by evaluating the alignment between its predictions and the actual labels during the testing phase.

---

### Result

The result section presents the performance metrics obtained during the testing phase. Typically, these performance metrics include accuracy, precision, recall, F1 score, and the confusion matrix, among others. These metrics provide insights into how well the model classifies and predicts. The result includes information about the model’s performance and accuracy on the test data.

---

## Code Description
### MLP.py
MLP.py defines the MLP model classes. The model class takes input size, hidden size, and output size as initialization parameters. `nn.Flatten()` flattens the input images into a 1D vector.

The `forward` method defines the forward pass of the model. It takes an input(‘X’) and first flattens it to 1D.

`linear_relu_stack` is composed of Sequential layers, including three linear layers and ReLU activation functions. This represents a 3-Layer MLP model with an input layer, a hidden layer, and an ouput layer. ReLU functions introduce non-linearity, allowing the model to learn complex patterns.

### TrainModel.py

TrainModel.py defines model and training parameters, including input size, hidden size, output size, training epochs, batch size, and learning rate. It loads the MNIST dataset and sets up data loaders.

Training epochs are performed a specified number of times `training_epochs`. The model is trained by iterating over mini-batches using a data loader `train_loader`. Loss is computed using `nn.CrossEntropyLoss()`, and backpropagation and optimization are performed.

### TestModel.py and TrainModel.py

The test dataset is used to evaluate the model. `model.eval()` is called to switch the model to evaluation mode, and the accuracy of the model is computed on the test data.

---
