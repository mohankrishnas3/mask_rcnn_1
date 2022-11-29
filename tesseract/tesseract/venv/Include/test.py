'REFERENCE LINKS'
'https://mlfromscratch.com/neural-network-tutorial/#/'
'https://towardsdatascience.com/an-introduction-to-neural-networks-with-implementation-from-scratch-using-python-da4b6a45c05b'
'https://medium.com/@zeeshanmulla/cost-activation-loss-function-neural-network-deep-learning-what-are-these-91167825a4de'

'import all necessary Packages'
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from random import sample

'The building block of the deep neural networks is called the sigmoid neuron. Sigmoid neurons are similar to perceptrons'
'but they are slightly modified such that the output from the sigmoid neuron is much smoother than the step functional output from perceptron. '
def sigmoid(x, derivative=False):
    if derivative:
        return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
    return 1 / (1 + np.exp(-x))
'Initilize all weights and Bias'
def initialize_params(layer_sizes):
    params = {}
    for i in range(1, len(layer_sizes)):
        params['W' + str(i)] = np.random.randn(layer_sizes[i], layer_sizes[i-1])*0.01
        params['B' + str(i)] = np.random.randn(layer_sizes[i],1)*0.01
    return params
'Forward Propagation is the way to move from the Input layer (left) to the Output layer (right) in the neural network'
def forward_propagation(X_train, params):
    layers = len(params)//2
    values = {}
    for i in range(1, layers+1):
        if i==1:
            values['Z' + str(i)] = np.dot(params['W' + str(i)], X_train) + params['B' + str(i)]
            values['A' + str(i)] = sigmoid(values['Z' + str(i)])
        else:
            values['Z' + str(i)] = np.dot(params['W' + str(i)], values['A' + str(i-1)]) + params['B' + str(i)]
            if i==layers:
                values['A' + str(i)] = values['Z' + str(i)]
            else:
                values['A' + str(i)] = sigmoid(values['Z' + str(i)])
    return values
'A cost function is a measure of error between what value your model predicts and what the value actually is'
def compute_cost(values, Y_train):
    layers = len((values))//2
    #print(layers)
    Y_pred = (values['A' + str(layers)])
    #print(Y_pred)
    cost = 1/(2*len(Y_train)) * np.sum(np.square(Y_pred - Y_train))
    return cost
'Backward Propagation is the preferable method of adjusting or correcting the weights to reach the minimized loss function'
def backward_propagation(params, values, X_train, Y_train):
    layers = len(params)//2
    m = len(Y_train)
    grads = {}
    for i in range(layers,0,-1):
        if i==layers:
            dA = 1/m * (values['A' + str(i)] - Y_train)
            dZ = dA
        else:
            dA = np.dot(params['W' + str(i+1)].T, dZ)
            dZ = np.multiply(dA, np.where(values['A' + str(i)]>=0, 1, 0))
        if i==1:
            grads['W' + str(i)] = 1/m * np.dot(dZ, X_train.T)
            grads['B' + str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        else:
            grads['W' + str(i)] = 1/m * np.dot(dZ,values['A' + str(i-1)].T)
            grads['B' + str(i)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    return grads
'update parameters after backpropogation'
def update_params(params, grads, learning_rate):
    layers = len(params)//2
    params_updated = {}
    for i in range(1,layers+1):

        params_updated['W' + str(i)] = params['W' + str(i)] - learning_rate * grads['W' + str(i)]
        params_updated['B' + str(i)] = params['B' + str(i)] - learning_rate * grads['B' + str(i)]
    return params_updated

x = []
x1 = []
'Train the model'
def model(X_train, Y_train, layer_sizes, n_batch,n_epoch, learning_rate):
    params = initialize_params(layer_sizes)

    for i in range(n_epoch):
        print('No of Epoch' +' = '+ str(i))
        train_acc, test_acc, train_acc1, test_acc1 = compute_accuracy(X_train, X_test, Y_train, Y_test,
                                                                      params)  # get training and test accuracy
        print('Mean Absolute Percentage Error on Training Data = ' + str(train_acc))
        print('Root Mean Squared Error on Training Data = ' + str(train_acc1))
        x.append(train_acc)
        x1.append(train_acc1)


        for j in range(n_batch):
            values = forward_propagation(X_train.T, params)
            cost = compute_cost(values, Y_train.T)
            grads = backward_propagation(params, values,X_train.T, Y_train.T)
            params = update_params(params, grads, learning_rate)
            print('Cost at iteration ' + str(j + 1) + ' = ' + str(cost) + '\n')

    return params
"Find regression model accuracy RMSE and MAPE"
def compute_accuracy(X_train, X_test, Y_train, Y_test, params):

         values_train = forward_propagation(X_train.T, params)
         values_test = forward_propagation(X_test.T, params)
         Y_pred_train = (values_train['A' + str(len(layer_sizes)-1)].T)
         Y_pred_test = (values_test['A' + str(len(layer_sizes) - 1)].T)
         collect = []
         collect1 = []
         for i in Y_pred_train:
             max_value = max(i)
             collect.append(max_value)
         for i in Y_pred_test:
             max_value = max(i)
             collect1.append(max_value)
         train_acc = mean_absolute_percentage_error((Y_train),collect)
         test_acc = mean_absolute_percentage_error((Y_test),collect1)
         train_acc1 = np.sqrt(mean_squared_error(Y_train, collect))
         test_acc1 = np.sqrt(mean_squared_error(Y_test, collect1))
         return train_acc, test_acc, train_acc1, test_acc1

'Predict Function to Predict valies on Custom Data'
def predict(X, params):
    values = forward_propagation(X.T, params)
    predictions = values['A' + str(len(values)//2)].T
    return predictions

'MAIN'
#data = pd.read_csv('winequality-red.csv')
data = pd.read_csv('wine.txt', sep=" ")
data= data.drop(columns='Unnamed: 12', axis=1)
actual_data = data.loc[:, data.columns != 'quality']
target = data["quality"]
max_layer = max(target)
print(max_layer)
actual_data = actual_data.to_numpy()
target = target.to_numpy()
X,Y = actual_data, target
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2)

layer_sizes = [11, 10 , 10]
print(Y_test)
print(len(Y_test))
n_batch = len(X_train)
n_epoch = 25
learning_rate = 0.8
print(len(X_train))
params = model(X_train, Y_train, layer_sizes, n_batch,n_epoch, learning_rate)
train_acc, test_acc, train_acc1, test_acc1 = compute_accuracy(X_train, X_test, Y_train, Y_test,
                                                              params)
print('Mean Absolute Percentage Error on Training Data = ' + str(train_acc))
print('Mean Absolute Percentage Error on Test Data = ' + str(test_acc))
print('Root Mean Squared Error on Training Data = ' + str(train_acc1))
print('Root Mean Squared Error on Test Data = ' + str(test_acc1))
#print(total)
# result = predict(X,params)
# print(result)
