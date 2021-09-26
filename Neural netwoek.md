<!-- Heading -->
<!-- ctr+shift+ c  -->
## Table of content 
### Neural Network 
##### The Perceptron
##### Weights and Biases
##### Activation Function/Error Function 
### Convolutional Neural Network 
##### Keras
#####  Convolution Layer 
##### Pooling 
##### Fully Connected Layers.
###  Techniques 
##### RELU Activation Function 
##### Flatten
##### Dropout
##### Onehot Encoding 
##### Loss Function
##### SoftMax
### CNN Implimentation
##### Pre - Processing Images
##### Image Tracking/Reference
##### Conversion to Grayscale 
##### Histogram equalization
##### Normalization/ Map function. 
##### Reshaping the data and Onehot Encoding it. 
##### Lenet model and Fine – Tuning
### Training the model 
### Testing 
---
--- 
## **Neural Network**
KITT will be using neural networks to classify images such as the road signs, KITT will be using this classification to instruct and control the movement of the vehicle. The network will be trained to classify different types of road signals, but the main purpose of those classifications is to make sure the car will understand when to stop. 
 
### **The Perceptron**
The perceptron is the most basic form of neural network, that takes inspiration from the brain. The perceptron like the brain is a basic processing unit gets inputs, processes it and predicts the output. Perceptron’s are trained to receive inputs and the form of input nodes and transfer the appropriate output, so in case of KITT these outputs will be to classify the road signs

 ![perceptron](https://github.com/penzen/Pictures/blob/main/Activation.png)

### **Weights and Biases**
The weights and biases are the basic units of the perceptron used to provide an input data processing that is used to decide how to categorize that data by producing an output.
 
 The generally equations for weights and biases is:

 w1(x1) + w2(x2) + b (Liner model)

When we start to train a model the weights and the biases are picked at random, and as the traverse through the data their variables change accordingly to reduce the error gradient (error function) of the data and give a accurate classification of the model.   


### **Activation function/Error function**
An activation function is used to get the output using the weights and biases, it uses the weight and bias to convert the output and depending on the output we get a error function, the error function will give us the an value that will be compared to the model, the error function can vary depending on the wights and biases.

![Activation](https://github.com/penzen/Pictures/blob/main/Activation.png)

## **Convolutional Neural Network**
Convolutional networks are a specialized form of artificial neural networks they are most effective in face recognition, object detection, and most notably for this project, identifying different traffic signs.

![Network](https://github.com/penzen/Pictures/blob/main/Network.png)

These networks are useful for analysing and classifying images because they are very effective at recognizing useful patterns within the images by understanding that spatial structure of the inputs is relevant.

### **Keras** 
Keras is a library that contains pre-existing tools to facilitate the construction of a neural network, to ensuring far more concise and easier code. Keras has numerous models that can be used to make and implement the neural network.

We can import Keras using the following line of code 

```
from tensorflow import keras
```
From Keras we import models we import Sequential.
```
from keras.models import Sequential
```
The Sequential model API is a way of creating deep learning models where an instance of the Sequential class is created, and model layers are created and added to it, the sequential model is the tool used to create our model. 

From Keras we will also need to import the dense layer, the dense layer is simply a group of connected neurons (perceptron) that will process the data and give an output. 
```
from keras.layers import Dense
```
Keras has Adam optimizer which will be used for the replacement optimization algorithm for stochastic gradient descent for training deep learning models it allows us to connect preceding layers in the network to subsequent layers, creating a fully connected layer network.
```
from  tensorflow.keras.optimizers import Adam
``` 

By claiming a variable equals to sequential will give us a empty model to build on, after making a variable sequential it then has access to all the functionality of the sequential.  
```
 model = Sequential()
```

Parameter 	Description
Learning rate(lr)	Lr is the short form for learning rate, the Adam optimizer will adjust the weights accordingly, the higher the learning rate faster the model learns.

|Parameters   | Description  | 
|---|---|
|Learning rate(lr)  | Lr is the short form for learning rate, the Adam optimizer will adjust the weights accordingly, the higher the learning rate faster the model learns.|


### **Convolution Layer.**
Convolutional neural network or CNN are computationally effective compared to another neural network. Convolutional neural network requires less input compared to a regular network. The name convolution comes from networks convolution operations which means, convolution network has three layers the convolutional layer, the pooling layers, and a fully connected layer. 

The convolution layer is the main building block of the convolutional network is the convolutional layer there. A primary goal of the convolution layer is to extract and learn specific image features, features that can be used to help classify the image. 

Inside the convolution layer the image is going to be proceed by kernel matrix, convolutional operation can be performed by sliding the kernel at every location of the image. The amount by which the kernel shifts at every operation is known as the Stride. 

![Kernel](https://github.com/penzen/Pictures/blob/main/Kernel.png)

The image is an example of a kernel matrix it. The values of the weights in the kernel are learned by the convolutional neural network during the training process through weight gradient descent algorithm, which acts to minimize the error function by changing the kernel values to the ones that are best able to detect features in the image.

The kernel matrix is multiplied with the receptive field (the area where the kernel multiplication takes place/ the image itself) to give us a feature map. The primary purpose of the convolutional layer is to extract and learn specific image features, features that will be used to help classify the image and the feature map contains a specific feature of interest which was extracted from the original image

```
model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
model.add(Conv2D(60, (5, 5), activation='relu'))
```
|Parameters   | Description  | 
|---|---|
|Conv2d  | The first input is the amount of filters(kernel matrix) we need, the second definition(5,5) is a tuple that specify the size of the filters|
|Input shape   | The shape of the image/ data that has been reconfigured; the shape of the image is 32 by 32 with a depth of 1(because of grayscale)  |   
|Activation   |  We use RELU function because it is a non – liner multiclassification activation function.   |  


To use conv2d we need to import it from Keras 
```
from keras.layers.convolutional import Conv2D
```
### **Pooling** 
The pooling layer acts to shrink the image stack by reducing the dimensionality of the representation of each feature map, thereby also reducing the computational complexity of the model. 

Although it retains the most important information such that the feature of interest is still consistent with its feature map, the image is scaled down because reduced computational costs, reduces the number of parameters in the image, and helps to reduce overfitting by providing an abstracted form of the original feature map.

Pooling operation is used in the pooling layer, they consist of 3 operations max, sum and average. Max Pulling provides a scale invariant representation of the image, which is very useful as it allows detect features in an image no matter where they are located. 

Pooling helps make the network remain unaffected by small translations in the input, image, or any distortion, because the one taking the maximum value in a local neighbourhood, the output of the pooling layer is still going to be very similar. This is convenient as it helps to generalize features to more than one image.

![Pooling](https://github.com/penzen/Pictures/blob/main/Pooling.png)

```
 model.add(MaxPooling2D(pool_size=(2, 2)))
```

|Parameters   | Description  | 
|---|---|
|pool_size|The maxpooling2d takes in one parameter which is the size of the pool, having 2 by 2 will result in our data becoming more abstract and scaled down while retaining all the information from the feature map.  |




To use the Max pooling operation, we need to import it from Keras 
```
from keras.layers.convolutional import MaxPooling2D
```

### **Fully connected Layers.** 
The fully connected layer is responsible for classification. 

The fully connected neural network is responsible in taking features (From the pooling layer) as inputs, processing them to obtain a final probability as to what class the image belongs to. Each neuron is connected to all the neurons in the previous layer and each connection has its own weight it simply updates its weights and by its values to minimize the total error function based on a gradient descent algorithm.

```
 model.add(Dense(500, activation='relu'))
```
|Parameters   | Description  | 
|---|---|
| Units | The amount of perceptron that will be used (500). |
| Activation   |  Choses which activation function to choose.  |   



### **Techniques**

#### **RELU Activation function.** 

 The rectified linear activation function or RELU is a function that converts all the negative values in the feature map to 0 and if the number is positive, it will output as it is. RELU function uses the outputs from the feature map which will give us the output of the image. The reason we are using the RELU function is because it solves the problem of vanishing gradients problem, which can be cause by the sigmoid function. 

 ```
  model.add(Conv2D(30, (3, 3), activation='relu'))
 ```

 |Parameters   | Description  | 
|---|---|
|Activation  |  RELU Activation funtion |

### **Softmax**
The SoftMax function is a activation function for scenarios involving multiclass functions, it will be used for classifying our different traffic signs. The SoftMax function keeps the magnitudes of the probability given by the perception of the different traffic signals, this allows us to accurately display a percentage of which categories the traffic sign falls within. 

To implement the SoftMax function we will use the following code. 
```
model.add(Dense(43, activation='softmax'))
```


|Parameters   | Description  | 
|---|---|
|(43) Output nodes| The first input takes in the amount of output nodes we need, in our case we have 43 different Traffic signs that need classification, so the variable num_classes simply equals to 43, giving us 43 different classification output nodes. |
| activation | Softmax multiclass classficiation functions  |   



### **Loss Function** 

Categorical cross-entropy distinguish between a good model and a bad one.

Categorical cross-entropy is a method of measuring error with any neural network, and consequently a lower cross entropy value implies a more accurate system, while a higher cross entropy value implies a less accurate system.

Categorical cross-entropy will allow us to distinguish how accurate our model predicts Traffic signs and accordingly adjust the weights and bias. 

We can use the categorical cross entropy with the following code.   
```
model.compile(Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
```
Parameters   | Description  | 
|---|---|
| Adam  | The first input is used for the optimizer, and we have used Adam for it.  |
| Loss  | Used to specify the loss function that will predict the error.  |  
| Metrics  | Metrics is used to calculate the specified, attribute(accuracy).  | 


### **Onehot encoding**

Onehot encoding makes sure our data sets are mutually exclusive meaning their label are independent of each other that is why we use one hot encoding for one hot encoding allows us to classify all our classes without assuming any dependence between the classes.

Onehot encoding works by creating separate columns for each data label in the data sets and using an appropriate one or a zero value for each column to identify the data class for our data.

We start by first importing the following from Keras.

```
from keras.utils.np_utils import to_categorical
```
```
Y_train = to_categorical(Y_train,43)
```

Parameters   | Description  | 
|---|---|
|To_categorical    | The first input takes the data that needs to be one_hot_encoded. |
| 43  |  The second parameter will be the number of classes we have, in our cases we want to classify 43 different traffic signs. |  


### **Dropout**
The drop out layer is used to prevent overfitting in the data, the way it works is by randomly setting a fraction rate of input units to zero at each update during training, which helps prevent overfitting, some nodes will be turned off and are no longer communicating information along the network, it reduces overfitting because it forces the neural network to use various combinations of nodes to classify the same data as with each update, a random set of nodes survives the dropout process.

Each node is forced to learn the data set in a more independent way with a unique neural arrangement each time.
This causes the neural network in general to become more versatile and allows the weights to be more uniformly distributed along the network.

```
 model.add(Dropout(0.5))
```
Parameters   | Description  | 
|---|---|
| Dropout() |The dropout layer has only one input, used to specify the fraction of the nodes that are going to be dropout, in our case it is 0.5(50%) which is the recommend amount    |


### **Flatten**

Flatten is properly format our data so it to go in the fully connected network, because the connected layer will need each image to be flatten to a one-dimensional array.


Parameters   | Description  | 
|---|---|
| Flatten()  | Flatten does not require any parameters as all it does is convert the data into a flat 1D array.  |



## **Pre- processing Images.**

 ## **Data Extraction**

The data/images used for this Neural network is from a repository that is composed of Traffic signs, we can use the following command and the link to clone the repository.  

Link: [Trafic signs](https://bitbucket.orgjadslim/german-traffic-signs)

```
!git clone https://bitbucket.org/jadslim/german-traffic-signs
```
We can use the following command to list the different files in the directory(folder)
```
!ls german-traffic-signs
```
Import pickle:
```
import pickle 
```
Pickled files in Python essentially contain serialized data that can be unpickled when desired. In our case the files we get from the repository are already picked, that means for the data to be usable the files will need be unpickled. 
```
signnames.csv  test.p  train.p	valid.p
```
We can unpickle those files using the following lines of code, remember to appropriately name the variable data to avoid confusion. 
```
with open('german-traffic-signs/train.p','rb') as f:
  train_data = pickle.load(f)
with open('german-traffic-signs/test.p','rb') as f:
  test_data = pickle.load(f)
with open('german-traffic-signs/valid.p','rb') as f:
  valid_data = pickle.load(f)
```
Import pandas 
```
import pandas as pd
```
Pandas is a software library written for the Python programming language for data manipulation and analysis. In our case we will be manipulating a CSV file which means a comma separated value files.

We will use read csv method in pandas to extract our data. 

```
data = pd.read_csv('german-traffic-signs/signnames.csv')
```

### **Image tracking/reference.**
 Pre- processing images makes it easier for the network to classify images, by pre-processing data the network will use less computational power when it’s processing the data. 

 To get started with the pre-processing we need to know the dimensions of the images (shape of the image/pixels), chose a random index in the data set and make sure it stays consistent because that data will be the hallmark of how our images have been transformed after the pre-processing. 

 ```
 import cv2 

plt.imshow(X_train[900])
plt.axis('off')
print(X_train[900].shape)
print(Y_train[900]) 
 ```

Parameters   | Description  | 
|---|---|
| Plt.imshow()  | The plt function is a function from the OpenCV library used to show the data, in our case it would be the image that’s located at the 900th index   |
| plt.axis()  |The off is used to turn off the grid, when the image is shown.    |  
| X_train[].shape               | The shape of the data in our case is extracted because it will be used for pre-processing   |

### **Conversion to Gray scale.**

The first being to convert our image to greyscale will define a function and death called and greyscale, which one invoked receives an RGB image. This will be a very simple function which converts an RGB image into a grayscale image.

```
def grayscale(img):
  img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # converts to gray 
  return img
```
Parameters   | Description  | 
|---|---|
| Cv2.cvtColor   | The cvtColor takes in two parameters, the data(image) and the type of conversion in our case it would be to gray using COLOR_BGR2GRAY   |

The greyscale conversion is done because, colour does not play an important role   distinguishing between traffic signs. The lighting in the image varies and many of the traffic signs of similar colours reinforcing that not a very relevant piece of information, the features of the traffic signs that really matter are the edges of the curves, the shape inside of it. That's what the network should focus on. 

The conversion of RGB to grayscale, reduces the depth of our image from three to one. This means that the network now requires fewer parameters as the input data will only have a depth of one channel in the long run. This means that our network will be much more efficient and will require less computing power to classify our data.

### **Histogram equalization**

Histogram equalization is a technique that aims to standardize the lighting and all the images. Some of the images are very bright and others very dim after histogram equalization, these images will have a similar lighting effect.

The process enhances the contrast in the image such that any grayscale intensities are now better distributed across the image, at the same time deemphasizing any pixel intensities that occur at high frequencies, this process also results in higher contrast within our image, which can help with feature extraction.

```
def equalize(img):
  img =cv2.equalizeHist(img) # equalizeHist only excepts grayscale images 
  return img 
```
Parameters   | Description  | 
|---         |---           | 
| Cv2.equalizeHist()  | The function takes in a single argument, which is simply the image that is being modified, and equalizes the pixel intensities.    | 

 Note : The equalize hist function will only accept a grayscale images as they don't have a depth as RGB images have three colour channels.



### **Normalization/ Map function.**  
To Normalize the image, the image needs to be divided all the pixel intensities of the image by two fifty-five. This causes all the pixel values in our image to be normalized between zero and one, this is done simply by taking our image and dividing it by two fifty-five.

```
def preprossing(img):
  img =grayscale(img)
  img =equalize(img)
  img =img/255 #normilization
  return img 

```
Parameters   | Description  | 
|---|---|
| Grayscale()  |The previously defined function grayscale will convert our image to grayscale for the next step.   | 
| Equalize()  |  After grayscale is done the image will be equalized using the histogram in this function.  | 
| Img/ 255    | The finally step is to normalize the images into binary ones and zeros by dividing the image by 255.  |

After the images have been pre-processed, we use a function called map, which will return the processed images as a list to be stored into their respective variables. 

```
X_train = np.array(list(map(preprossing, X_train))) 
X_val = np.array(list(map(preprossing, X_val))) 
X_test = np.array(list(map(preprossing, X_test))) 

```

### **Reshaping the data and onehot encoding it.** 
 The way convolutional networks work is by applying a filter to the channels of the image that's being viewed in the case of grayscale images.
There is one channel present there for our data must reflect the presence of the steps
and so, by adding this depth, our data will be in the desired shape to be used as an input for the convolutional layer.

```
Y_train = to_categorical(Y_train, 43)
Y_test = to_categorical(Y_test, 43)
Y_val = to_categorical(Y_val, 43)
```
```
X_train = X_train.reshape(34799, 32, 32,1)
X_test = X_test.reshape(12630, 32, 32,1)
X_val = X_val.reshape(4410, 32, 32,1)
```

### ***Lenet model and fine – tuning***

The Lenet architecture is convolution neural network model consisting of the following structure, The first layer of the network is a convolutional layer with an unspecified amount of filter is the output of this convolutional layer is then fed into a pooling layer. 

That layer is then connected to another convolutional layer and finally, we have one more pooling layer before our data is fed into a fully connected layer that eventually connects to an output classifier.

![Lenet Model](https://github.com/penzen/Pictures/blob/main/Lenet.png)

The architecture of this model is inspired by the Lenet model but is modified to fine tune the model, as the original Lenet model was not efficient enough to classify the images with high accuracy. 

The image below is the original lenet model used to train the data, but it seemed that this model was not enough to accurately classify images. 

```
def lenet_model():
  model = Sequential()
  model.add(Conv2D(30, (5,5),input_shape =(28,28,1),activation='relu',))# adds the convolution layer 
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Conv2D(15, (3,3),activation='relu',))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Flatten()) 
  model.add(Dense(500,activation='relu'))
  model.add(Dropout(rate = 0.5 ))
  model.add(Dense(num_classes, activation='softmax')) 
  model.compile(Adam(learning_rate= 0.01), loss='categorical_crossentropy', metrics=['accuracy'])
  return model
The modified model has decreased Adam learning rate meaning it’s going to take more time training the data as the learning rate has decreased by a factor of 10, this means the accuracy will improve the accuracy of the data. 
```
Before:
```
model.compile(Adam(learning_rate= 0.01), 
```
After:
```
model.compile(Adam(learning_rate= 0.001), 
```

The convolutional layer has also been modified increasing the numbers of filter to 60 instead of 30, meaning there are going to be more parameters as because the filter is twice the size, this will help improve the overall accuracy of the model. 
 
Before: 
```
model.add(Conv2D(30, (5,5),input_shape =(28,28,1),
model.add(Conv2D(15, (3,3),input_shape =(28,28,1),
```
After:
```
model.add(Conv2D(60, (5,5),input_shape =(28,28,1),
model.add(Conv2D(30, (5,5),input_shape =(28,28,1),
```
The modifed model has two  additional convolutional layers and another dropout layer now we have 4 convolutional layers, meaning this will help the network extract features more accurately.
```
model.add(Conv2D(60, (5,5),input_shape =(28,28,1)
model.add(Conv2D(30, (5,5),input_shape =(28,28,1)
model.add(Dropout(rate = 0.5 ))
```
The final version of the modified Lenet model 
```
def modified_model():
  model = Sequential()
  model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
  model.add(Conv2D(60, (5, 5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(43, activation='softmax'))
  
  model.compile(Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
  return model
```

## **Training** 
To train our model, we need to use a function called fit, fit has a few parameters

```
history = model.fit(X_train,Y_train, epochs= 10, validation_data=(X_val, Y_val), batch_size= 400,verbose = 1, shuffle= 1 ) train the model without the fit generator 
```
Parameters   | Description  | 
|---|---|
|  X_train and  y_train  | The first two inputs are the data sets we use to train the model with.   | 
| Epochs   | Epochs is the number of times the neural network goes through the dataset  | 
|Validation_data    | Validation data is the data used against the training set to validate the accuracy of the model  |
|  Batch_size  | Batch size is the amount of data the neural network works with per iteration  | 
| Shuffle   | Shuffle helps the model classify images better by  shuffling the dataset at each epoch   |


## **Testing**
Testing is an essential part of improving the neural network, testing tells us if the neural network has in fact been trained properly or not.

We import a random image of a traffic sign and use it to test our network.

Link used in the demo: [demo_image ](https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg)

```
import requests
from PIL import Image
url = 'https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg'
r = requests.get(url, stream=True)
img = Image.open(r.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))
```

Before testing the image we will need to pre-process it as it will not work as it is because the neural network will not work with the parameters and the size of the image. 

```
#Preprocess image
img = np.asarray(img)
img = cv2.resize(img, (32, 32)) #because  we trained our model on 32/32 
img = preprossing(img)
plt.imshow(img, cmap = plt.get_cmap('gray'))
print(img.shape)
```
We reshape the image to input it for the predication, and then use model.predict to predict the results. 
```
img = img.reshape(1, 32, 32, 1)
print("predicted sign: "+ str(model.predict(img)))
```
Parameters   | Description  | 
|---|---|
|Model. Predict()    |  Model. Predict needs an image as a parameter, so it can predict what kind of image it actually is.  |