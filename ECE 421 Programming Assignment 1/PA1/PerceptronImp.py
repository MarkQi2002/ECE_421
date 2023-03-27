import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 

def fit_perceptron(X_train, y_train):
    # Input:
    # X_train represents the matrix of input features that belongs to R (N By d)
    # N is the total number of training samples
    # d is the dimension of each input feature vector

    # y_train is an N dimensional vector where the ith component represents 
    # the output observed in the training set for the ith row in X_train matrix
    # which corresponds to the ith input feature (+1 or -1)

    # Output: 
    # Vector w that represents the coefficients of the line computed by the pocket algorithm 
    # that best separates the two classes of training data points
    # Vector w has dimensions d + 1

    # Variable Declaration And Intialize Weight Vector
    w = np.zeros(X_train.shape[1] + 1)

    # Number Of Epoch
    epochs = 0

    # Add Additional Column To The Front Of X_train
    one_column = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((one_column, X_train))

    # Perceptron Learning Algorithm (PLA)
    while (errorPer(X_train, y_train, w) != 0 and epochs < 5000):
        # Increment Epochs
        epochs += 1

        # Iterate Through All Data Points
        for i in range(X_train.shape[0]):
            # Modify Weight Vector Based On Misclassified Data Points
            if (pred(X_train[i], w) != y_train[i]):
                w = w + y_train[i] * X_train[i]

    # Return Trained Weight Vector
    return w

def errorPer(X_train, y_train, w):
    # Input:
    # X_train represents the matrix of input features that belongs to R (N By d)
    # N is the total number of training samples
    # d is the dimension of each input feature vector

    # y_train is an N dimensional vector where the ith component represents 
    # the output observed in the training set for the ith row in X_train matrix
    # which corresponds to the ith input feature (+1 or -1)
    
    # w is the weight vector of d + 1 dimensions

    # Output:
    # Error Percentage

    # Variable Declaration
    error_number = 0
    error_percentage = 0

    # Iterate Through All Input Features
    for i in range(X_train.shape[0]):
        if (pred(X_train[i], w) != y_train[i]):
            error_number += 1
    
    # Calcuate Actual Error Perceentage
    error_percentage = error_number / X_train.shape[0]

    # Return Error Percentage
    if (error_number == 0):
        return 0
    else: 
        return error_percentage

def confMatrix(X_train, y_train, w):
    # Input:
    # X_train represents the matrix of input features that belongs to R (N By d)
    # N is the total number of training samples
    # d is the dimension of each input feature vector

    # y_train is an N dimensional vector where the ith component represents 
    # the output observed in the training set for the ith row in X_train matrix
    # which corresponds to the ith input feature (+1 or -1)
    
    # w is the weight vector of d + 1 dimensions

    # Output:
    # A two-by-two matrix composed of integer values

    # Variable Declaration
    confusion_matrix = np.zeros((2, 2), np.int64)

    # Add Additional Column To The Front Of X_train
    one_column = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((one_column, X_train))

    # Iterate Through All Input Features
    for i in range(X_train.shape[0]):
        if (pred(X_train[i], w) == 1):
            # True Positive
            if (y_train[i] == 1):
                confusion_matrix[1][1] += 1
            # False Positive
            else:
                confusion_matrix[1][0] += 1
        else:
            # False Negative
            if (y_train[i] == 1):
                confusion_matrix[0][1] += 1
            # True Negative
            else:
                confusion_matrix[0][0] += 1
    
    # Return Final Confusion Matrix
    return confusion_matrix


def pred(X_i, w):
    # Input:
    # X_i is the feature vector of d + 1 dimensions of the ith test datapoint
    # w is the weight vector of d + 1 dimensions

    # Output:
    # Class predicted for linear classifier defined by w for the input data point X_i
    # Either +1 or -1
    
    # Dot Product
    predicted_value = np.dot(X_i, w)

    # Strictly Positive (+1) Otherwise (-1)
    if (predicted_value > 0):
        return 1
    else:
        return -1


def test_SciKit(X_train, X_test, Y_train, Y_test):
    # Pocket Algorithm Using scikit-learn Model
    # Training Weight Vector Using scikit-learn Model
    pct = Perceptron()
    pct.fit(X_train, Y_train)

    # Pass In The Test Features Into The Trained Model
    pred_pct = pct.predict(X_test)

    # Return Confusion Matrix
    return confusion_matrix(Y_test, pred_pct)

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:], y_train[50:], test_size=0.2)

    #Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    #Pocket algorithm using Numpy
    w = fit_perceptron(X_train,y_train)
    cM = confMatrix(X_test,y_test,w)

    #Pocket algorithm using scikit-learn
    sciKit = test_SciKit(X_train, X_test, y_train, y_test)
    
    #Print the result
    print ('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()
