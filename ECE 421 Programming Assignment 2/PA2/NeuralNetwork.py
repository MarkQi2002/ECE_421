import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 

# Part 2 Function Implementation
def fit_NeuralNetwork(X_train, y_train, alpha, hidden_layer_sizes, epochs):
    # Initialize Epoch Errors To Zero Vector
    err = np.zeros((epochs, 1))
    
    # Initialize Architecture
    N, d = X_train.shape
    X0 = np.ones((N, 1))
    X_train = np.hstack((X0, X_train))
    d = d + 1
    L = len(hidden_layer_sizes)
    L = L + 2
    
    # Initialize Randomized Weight For Input Layer
    weight_layer = np.random.normal(0, 0.1, (d,hidden_layer_sizes[0]))
    weights = []
    weights.append(weight_layer)
    
    # Initialize Weight For Hidden Layers
    # Iterate Through Hidden Layers
    for l in range(L - 3):
        weight_layer = np.random.normal(0, 0.1, (hidden_layer_sizes[l] + 1, hidden_layer_sizes[l + 1])) 
        weights.append(weight_layer) 

    # Initialize Weights For Output Layer
    weight_layer = np.random.normal(0, 0.1, (hidden_layer_sizes[l + 1] + 1, 1)) 
    weights.append(weight_layer) 
    
    # Iterate Through Epochs
    for e in range(epochs):
        # Stochastic Gradient Descent Choice
        choiceArray = np.arange(0, N)
        np.random.shuffle(choiceArray)
        errN = 0

        # Iterate Through Samples
        for n in range(N):
            # Selected Sample Variable Declaration
            index = choiceArray[n]
            x = np.transpose(X_train[index])
            y = y_train[index]
            
            # Forward Propagation, Backward Propagation, Update Weight, Calculate Error
            retX, retS = forwardPropagation(x, weights)
            g = backPropagation(retX, y, retS, weights)
            weights = updateWeights(weights, g, alpha)
            errN += errorPerSample(retX, y)

        # Record Average Error In Epoch e
        err[e] = errN / N 
    
    # Return Output
    return err, weights

# Part 2 Function Implementation
def forwardPropagation(x, weights):
    # Variable Declaration
    l = len(weights) + 1
    currX = x
    retS = []
    retX = []
    retX.append(currX)

    # Forward Propagate For Each Layer
    for i in range(l - 1):
        currS = np.dot(currX, weights[i])
        retS.append(currS)
        currX = currS
        
        # Hidden Layer
        if i != len(weights) - 1:
            # Iterate Through Nodes In Layer i
            for j in range(len(currS)):
                currX[j]= activation(currS[j])
            currX = np.hstack((1, currX))
        # Output Layer
        else:
            currX = outputf(currS)

        # Append Result
        retX.append(currX)

    # Return Output
    return retX, retS

# Part 2 Function Implementation
def errorPerSample(X, y_n):
    # Log Loss Error Function
    e_n = errorf(X[-1], y_n)

    # Return Output
    return e_n

# Part 2 Function Implementation
def backPropagation(X, y_n, s, weights):
    # x: 0, 1, ..., L
    # S: 1, ..., L
    # weights: 1, ..., L
    l = len(X)
    delL = []

    # To be able to complete this function, you need to understand this line below
    # In this line, we are computing the derivative of the Loss function w.r.t the 
    # output layer (without activation). This is dL/dS[l-2]
    # By chain rule, dL/dS[l-2] = dL/dy * dy/dS[l-2] . Now dL/dy is the derivative Error and 
    # dy/dS[l-2]  is the derivative output.
    delL.insert(0, derivativeError(X[l - 1], y_n) * derivativeOutput(s[l - 2]))
    curr = 0
    
    # Now, let's calculate dL/dS[l-2], dL/dS[l-3],...
    for i in range(len(X) - 2, 0, -1): # L-1, ..., 0
        delNextLayer = delL[curr]
        WeightsNextLayer = weights[i]
        sCurrLayer = s[i - 1]
        
        # Init this to 0s vector
        delN = np.zeros((len(s[i - 1]), 1))

        # Now We Calculate The Gradient Backward
        # Remember: dL/dS[i] = dL/dS[i+1] * W(which W???) * activation
        for j in range(len(s[i - 1])): # Number Of Nodes In Layer i - 1
            for k in range(len(s[i])): # Number Of Nodes In Layer i
                delN[j] = delN[j] + delNextLayer[k] * WeightsNextLayer[j, k] * derivativeActivation(sCurrLayer[j])
        
        delL.insert(0, delN)
    
    # We have all the deltas we need. Now, we need to find dL/dW.
    # It's very simple now, dL/dW = dL/dS * dS/dW = dL/dS * X
    g = []
    for i in range(len(delL)):
        rows, cols = weights[i].shape
        gL = np.zeros((rows,cols))
        currX = X[i]
        currdelL = delL[i]
        for j in range(rows):
            for k in range(cols):
                # Calculate Gradient Using currX and currdelL
                gL[j, k] = currX[j] * currdelL[k]
        g.append(gL)

    # Return Output
    return g

# Part 2 Function Implementation
def updateWeights(weights, g, alpha):
    nW = []
    for i in range(len(weights)):
        rows, cols = weights[i].shape
        currWeight = weights[i]
        currG = g[i]
        for j in range(rows):
            for k in range(cols):
                # Gradient Descent Update
                currWeight[j, k] = currWeight[j, k] - alpha * currG[j, k]
        nW.append(currWeight)
    
    # Return Output
    return nW

# Part 1 Function Implementation
def activation(s):
    # ReLU Activation Function
    x = max(0, s)

    # Return Output
    return x

# Part 1 Function Implementation
def derivativeActivation(s):
    # ReLU Activation Function Derivative
    # When Input Greater Than Zero, Derivative Is One, Otherwise Is Zero
    if (s > 0):
        x_L = 1
    else:
        x_L = 0
    
    # Return Output
    return x_L

# Part 1 Function Implementation
def outputf(s):
    # Logistic Regression (Sigmoid) Function
    x_L = 1 / (1 + np.exp(-s))

    # Return Output
    return x_L

# Part 1 Function Implementation
def derivativeOutput(s):
    # Logistic Regression (Sigmoid) Function Derivative
    x_L = np.exp(-s) / ((1 + np.exp(-s) ** 2))

    # Return Output
    return x_L

# Part 1 Function Implementation
def errorf(x_L, y):
    # Log Loss Error Function
    if (y == 1):
        e_n = -np.log(x_L)
    else:
        e_n = -np.log(1 - x_L)
    
    # Return Output
    return e_n

# Part 1 Function Implementation
def derivativeError(x_L, y):
    # Log Loss Error Function Derivative
    if (y == 1):
        e_n = (-1) / x_L
    else:
        e_n = 1 / (1 - x_L)
    
    # Return Output
    return e_n

# Part 3 Function Implementation
def pred(x_n, weights):
    # Run Forward Propagation With Given Weights
    retX, retS = forwardPropagation(x_n, weights)

    # Determine Output Label With Threshold 0.5
    if (retX[-1] >= 0.5):
        return 1
    else:
        return -1

# Part 3 Function Implementation
def confMatrix(X_train, y_train, w):
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

# Part 3 Function Implementation
def plotErr(e, epochs):
    # Plotting Training Error Function
    plt.title("Training Curve")
    plt.plot(e, label = "Train", linewidth = 2.0)
    plt.xlabel("Epochs")
    plt.ylabel("Trainig Error")
    plt.show()

# Part 3 Function Implementation
def test_SciKit(X_train, X_test, Y_train, Y_test):
    # SciKit Learn Neural Network
    NN = MLPClassifier(solver = 'adam', alpha = 0.00001, hidden_layer_sizes = (30, 10), random_state = 1)
    NN.fit(X_train, Y_train)

    # Testing
    NN_pred = NN.predict(X_test)

    # Return Output
    return confusion_matrix(Y_test, NN_pred)

# Provided Code
def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:], y_train[50:], test_size = 0.2, random_state = 1)
    
    for i in range(80):
        if y_train[i] == 1:
            y_train[i] = -1
        else:
            y_train[i] = 1
    for j in range(20):
        if y_test[j] == 1:
            y_test[j] = -1
        else:
            y_test[j] = 1
        
    err, w = fit_NeuralNetwork(X_train, y_train, 1e-2, [30, 10], 100)
    
    plotErr(err, 100)
    
    cM = confMatrix(X_test, y_test, w)
    
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ", cM)
    print("Confusion Matrix from Part 1b is:", sciKit)

test_Part1()

# Part 3 Question
def test_Part3():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:], y_train[50:], test_size = 0.2, random_state = 1)
    
    for i in range(80):
        if y_train[i] == 1:
            y_train[i] = -1
        else:
            y_train[i] = 1
    for j in range(20):
        if y_test[j] == 1:
            y_test[j] = -1
        else:
            y_test[j] = 1
        
    list = [(5, 5), (10, 10), (30, 10)]
    for item in list:
      NN = MLPClassifier(solver='adam', alpha=0.00001, hidden_layer_sizes=item, random_state=1)
      NN.fit(X_train, y_train)

      NN_pred = NN.predict(X_test)
      sciKit = confusion_matrix(y_test, NN_pred)
      NN_pred = NN.predict(X_train)
      scikkit = confusion_matrix(y_train, NN_pred)
      print("For ", item)
      print("Confusion Matrix for train data from Part 3 question is", scikkit)
      print("Confusion Matrix for test data from Part 3 question is:", sciKit)

test_Part3()
