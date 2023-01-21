import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
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

    # Add Additional Column To The Front Of X_train
    one_column = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((one_column, X_train))

    # Compute Pseudo-Inversion
    X_train_transposed = np.transpose(X_train)
    pseudo_inverse = np.matmul(np.linalg.pinv(np.matmul(X_train_transposed, X_train)), X_train_transposed)

    # Return Weight Vector
    return np.matmul(pseudo_inverse, y_train)


def mse(X_train, y_train, w):
    # Variable Declaration
    result = 0

    # Add Additional Column To The Front Of X_train
    one_column = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((one_column, X_train))

    # Output Mean Squared Error
    for i in range(X_train.shape[0]):
        result += pow(pred(X_train[0], w) - y_train[i], 2)

    # Divide By Total Number Of Features
    result = result / X_train.shape[0]

    # Return Mean Squared Error
    return result

def pred(X_train, w):
    # Predicted Value Based On Weight Vector
    return np.dot(X_train, w)

def test_SciKit(X_train, X_test, Y_train, Y_test):
    # Training Linear Regression Model From scikit-learn
    LR = linear_model.LinearRegression().fit(X_train, Y_train)

    # Calculate Mean Squared Error From scikit-learn
    lr_err = mean_squared_error(LR.predict(X_test), Y_test)

    # Return Final Mean Squared Error Rate
    return lr_err

def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    
    w = fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e = mse(X_test, y_test, w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
testFn_Part2()

# Mean Squared Error Computed From Our Own Linear Regression Model Were 7,035
# Mean Squared Error Computed From scikit-learn Library Were 3,249
# The Difference Were Not A Lot, So Our Model Performed Well