# Alphabet Classifier

Image classifier for three alphabets(A,T,O), using multilayer perceptron model. 

## Getting Started

#### Prerequisites

OpenCV is required to generate inpute vectors for training and testing. *How to install-*
```
pip install opencv-python
```

#### The Dataset
The dataset contains (15x100) binary matrix, made by flattening (10x10) alphabet [images](/inputs). There is pre-generated dataset, **"[dataset.npy](dataset.npy)"** and along with it **"[targets.npy](targets.npy)"** for 15 images that are inside "/inputs".

#### Generating the dataset
Generating the dataset for any three alphabets:
```
[create_dataset.py](create_dataset.py)
images, targets = generate(folderpath)
```
images is (numofimages x 100) matrix
targets is (number of images x 3 x 1) matrix for 3 alphabets

### Training the network
[trainmodel1.py](trainmodel1.py) and [trainmodel2.py](trainmodel2.py) are used to train 1 and 2 hidden layer perceptron model.

*[trainmodel1.py](trainmodel1.py)* - Contains (100 x 50 x 3) nodes, 100 being the input layer of feature vectors, hidden layer containing 50 nodes, and 3 output nodes generating 1 for every particular image, for my example-
A=\[1,0,0]
O=\[0,0,1]
T=\[0,1,0]
It generates weight matrices (100 x 50), (50 x3) and two bias vectors (1 x 50), (1 x 3). 

*[trainmodel2.py](trainmodel2.py)* - Contains (100 x 50 x 16 x 3) nodes, 100 being the input layer of feature vectors, hidden layer containing 50 nodes and 16 nodes, and 3 output nodes generating 1 for every particular image.
It generates weight matrices (100 x 50), (50 x16), (16 x 3) and three bias vectors (1 x 50), (1 x 16), (1 x 3).

**Activation function used for every node is sigmoid.**  
**Learning rate = 0.001**  
**Epochs=10^6**  
**Error calculated using Mean Squared Error**  

The above outputs are stored in "inputs#", where # is model number.

## Running the tests
Use [main.py](main.py)

### Getting weights

Loading weights and target:
```
weights, bias= get_model(model_number)
```

### Getting testset

Randomly an alphabet [image](testset) is selected from ["/folderpath"](testset) and feature vector(1 x 100) is generated for it. 

```
feature_vec, image_name=get_testset("folderpath")
```
### Running the model
output is (1 x 3) vector:
```
output=run_model(modelnumber, weights, bias, input)
```
The output generated and corresponding image_name is printed as output.

A=[1,0,0]
O=[0,0,1]
T=[0,1,0]
These are the ideal values for images of alphabets A, T, O.

### Example outputs
##### MODEL1

|![](https://i.imgur.com/kVh9Wsn.jpg)|![](https://i.imgur.com/JvWueen.png)|
| -------- | -------- |
|![](https://i.imgur.com/ZUkMShB.jpg)|![](https://i.imgur.com/mR1ULWS.png)|
|![](https://i.imgur.com/oN4VeEo.jpg)|![](https://i.imgur.com/bHnmPBf.png)|
|![](https://i.imgur.com/SbNgg56.jpg)|![](https://i.imgur.com/SvrmRuF.png)|
|![](https://i.imgur.com/F3rgGZV.jpg)|![](https://i.imgur.com/pvfexwQ.png)|
|![](https://i.imgur.com/BJvBRO7.jpg)|![](https://i.imgur.com/4Mk57cV.png)|

#### MODEL2

|![](https://i.imgur.com/TlMK9tU.jpg)|![](https://i.imgur.com/3RS0cHs.png)|
| -------- | -------- |
|![](https://i.imgur.com/Ut5Z2y0.jpg)|![](https://i.imgur.com/R2KSDHq.png)|
|![](https://i.imgur.com/vSGztMD.jpg)|![](https://i.imgur.com/XktuzBr.png)|
|![](https://i.imgur.com/DSwzYHA.jpg)|![](https://i.imgur.com/BHq0e4x.png)|
|![](https://i.imgur.com/380JRy6.jpg)|![](https://i.imgur.com/SnBgj31.png)|

## Built With

* [OpenCV](https://opencv.org/) - Image operations.
* [Maven](https://numpy.org/) - Matrix operations.

## Authors

* **Abhishek Kumar Yadav** - *Initial work* - [abhk943](https://github.com/abhk943)


## Acknowledgments

* [NNandDeepLearning](http://neuralnetworksanddeeplearning.com/chap2.html) - Contains a simple article on error back propagation.
