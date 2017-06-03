This is based on the content provided [here](https://www.udemy.com/deeplearning/learn/v4/overview) and simplified with comments for personal understanding.
This is a simple ANN which predicts if a customer will leave the bank or not.

## Data loading
We will use Pandas to load the data in csv


```python
import pandas as pd
dataset = pd.read_csv("data.csv")

#We are taking only the independent variables which impact the dependent variable (i.e. if the customer leaves the banks)
X = dataset.iloc[:,3:13].values

y = dataset.iloc[:,13].values
```

## Encoding
Now, we need to encode categorical data, as in mathematical equations we need to use only numerical values.
We encode all the independent variables which are non numeric.
In our current case, we need to encode ```Geography``` and ```Gender```


```python
# Encoding categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1]= labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2]= labelencoder_X_2.fit_transform(X[:,2])

```

We need to include dummy variables to avoid un-necessary preference to any encoded value as all the variables are of same importance.


```python
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
```

### Avoid dummy variable trap
We need to avoid dummy variable trap. More info on dummy variable trap is [here](http://www.algosome.com/articles/dummy-variable-trap-regression.html)


```python
#Done to avoid Dummy variable trap. Reducing dummy variable by one
X = X[:,1:]
```

## Data split
Splitting the data in training and test set


```python
#Splitting the dataset into the training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

```

## Feature Scaling
Feature scaling is done to avoid dominance of one independent variable on others.
We need to fit and transform the training dataset.
But test dataset is only transformmed.


```python
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Artificial Neural Network
Now we will create ANN with following approach
- Randomly initialise the weights to small numbers close to 0 (but not 0)
- Input the first observation of your dataset in the input layer, each feature in one input node.
- Forward-Propagation from left to right, the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate the activations until getting the predicted result y.
- Compare the predicted result to the actual result. Measure the generated error.
- Back-Progpagation: from right to left, the error is back-propagated so that the weights are updated accordingly.
- Repeat Steps 1-5 till whole data set is exhausted.


```python
#ANN 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
classifier = Sequential()

#Adding the input layer
#Number of nodes in hiddedn layer = average of number of nodes in input layer and number of nodes in output layer
#It is recommended to use RELU for hidden layers and Sigmoid (for two categories)/Softmax (for more than two categories) for output layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.1))
#Adding second hidden layer 
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
#Adding Dropout to avoid overfitting
classifier.add(Dropout(rate=0.1))
#Adding output layer 
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
# If dependent variable is binary then loss = binary_crossentropy else loss = categorical_crossentropy
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=10,epochs=100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

```

    Epoch 1/100
    8000/8000 [==============================] - 2s - loss: 0.4904 - acc: 0.7954     
    Epoch 2/100
    8000/8000 [==============================] - 2s - loss: 0.4357 - acc: 0.7960     
    Epoch 3/100
    8000/8000 [==============================] - 2s - loss: 0.4333 - acc: 0.7960     
    Epoch 4/100
    8000/8000 [==============================] - 2s - loss: 0.4330 - acc: 0.7960     
    Epoch 5/100
    8000/8000 [==============================] - 2s - loss: 0.4277 - acc: 0.7960     
    .
    .
    . 
    Epoch 95/100
    8000/8000 [==============================] - 2s - loss: 0.4204 - acc: 0.8295     
    Epoch 96/100
    8000/8000 [==============================] - 2s - loss: 0.4206 - acc: 0.8306     
    Epoch 97/100
    8000/8000 [==============================] - 2s - loss: 0.4219 - acc: 0.8325     
    Epoch 98/100
    8000/8000 [==============================] - 2s - loss: 0.4209 - acc: 0.8305     
    Epoch 99/100
    8000/8000 [==============================] - 2s - loss: 0.4194 - acc: 0.8304     
    Epoch 100/100
    8000/8000 [==============================] - 2s - loss: 0.4206 - acc: 0.8300     

```python
import numpy as np
new_prediction = classifier.predict(sc.transform(np.array([[0.,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction=(new_prediction>0.5)
print("The customer will leave" if new_prediction else"The customer will not leave")
```

    The customer will not leave


This is a simple ANN to determine if a customer will leave the bank or not.

