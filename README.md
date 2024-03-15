# Hand written character and digit recognition 
This project implements a Convolutional Neural Network (CNN) for recognizing handwritten characters and digits.

# dataset
It combines the MNIST dataset for digit recognition and a custom dataset for character recognition from kaggle.
you can download character dataset from:https://www.kaggle.com/datasets?search=a-z+characters+dataset

MNIST data set contains 70000 didgits data of 28*28 pixels format which can be loaded using mnist.load_data()
and character datadest contains 372450 characters data in csv file in which each image is flattened into row of 784(28*28) pixels

# Model Architechure
CNN model consists of total 5 layers in which 2 are convolution layers and other two are fully connected dense layer

Input Layer: The input layer defines the shape of the input data. In this model, the input shape is (28, 28, 1), indicating grayscale images of 28x28 pixels.
Convolutional Layers: Two convolutional layers are employed, each with 32 filters of size 5x5. These layers learn to extract features from the input images through convolutions.
Batch Normalization Layers: Batch normalization is applied after each convolutional layer to standardize the activations, stabilizing and accelerating the training process.
MaxPooling Layer: Following the second convolutional layer is a max-pooling layer with a pooling window of size 5x5. MaxPooling reduces spatial dimensions, aiding computational efficiency and preventing overfitting.
Dropout Layer: Dropout regularization is applied to reduce overfitting. A dropout rate of 0.25 is employed to randomly deactivate 25% of neurons during training.
Flatten Layer: The output from the last convolutional layer is flattened into a one-dimensional vector to prepare it for input into the fully connected layers.
Dense Layers: Two dense (fully connected) layers are included. The first dense layer has 256 units with ReLU activation, while the second layer has 36 units with softmax activation, representing the number of output classes (digits 0-9 and characters A-Z)
