import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import glob
import statistics
import math

def slope(m = 1, b = 0, min_x = 1, max_x = 1, x_num = 100):
    
    """""
    
    Outputs x and y values to plot for a line 
    
    Args:
        m = slope of the graph, 1 default
        b = y intercept, 0 default
        min_x = minimum x value
        max_x = maximum x value
        
    """""
    x_values = np.linspace(min_x, max_x, x_num) # x values
    y_values = m * x_values + b # y values 
    
    return x_values, y_values


def residual_scatter(x, y, m = 1, yintercept = 0, line = 'slope'):
    """Returns 1 graph and 1 residual amount for a set of x and y's

    Args:
        x (list): List of x values 
        y (list): List of y values
        m (int, optional): Starting slope. Defaults to 1.
        yintercept (int, optional): Starting y intercept. Defaults to 0.
        line (str, optional): Line type: either slope or flat. If flat, the y intercept is the average of y values. Defaults to 'slope'.

    Returns:
        List: First value is graph, second value is residuals amount
    """
    if line == 'slope':
        # Sum of squared residuals
        SSresiduals = sum([(y[i] - ((m * x[i]) + yintercept))**2 for i in range(len(x))])
    elif line == 'flat':
        yMean = statistics.mean(y) # Average
        # Sum of squared residuals
        SSresiduals = sum([(y_values - yMean)**2 for y_values in y])
    else:
        print("Choose either slope or flat")

    # Predicted y values for residuals line
    yPred = [m * x[i] + yintercept for i in range(len(x))]

    # Plotting scatterplot 
    plt.figure(figsize = (9,8)) # Bigger graph
    plt.scatter(x,y) # Scatterplot with x,y values

    # Plotting slope
    xSlope, ySlope = slope(m = m, b = yintercept, min_x = min(x), max_x = max(x)) # Getting x and y values
    plt.plot(xSlope, ySlope, 'r-') 

    # Plotting residuals
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], yPred[i]], 'g--') # Veritcal lines 

    # plt.ylim(bottom = 14) # Start at 14
    # plt.xlim(left = 0.6) # Start at 0.6
    
    SSresiduals = math.ceil(SSresiduals) # Rounding up numbers
    m = round(m, 4)
    yintercept = round(yintercept, 4)
    
    plt.title(f"\nResidual: $\\bf{SSresiduals}$\nSlope: {m}\ny-intercept: {yintercept}") # Graph title
    plt.show()
    
    return plt, SSresiduals


def gradient(x, y, m, b, learning_rate = 0.01):
    """Returns slope and y intercept of 1 iteration

    Args:
        x (list / np array): x values
        y (list / np array): y values
        m (int): slope
        b (int): y intercept
        learning_rate (float, optional): Smaller learning_rate means more steps ie more epochs to run before hitting learning_rate. Defaults to 0.01.

    Returns:
        List: First value is graph, second value is residuals
    """
    # Initialize derivative slope and y intercept at 0
    db = 0.0
    dm = 0.0
    n = len(x) # Number of coordinates
    
    # Sum of squared residuals of derivatives with respect to slope and y intercept
    for x, y in zip(x, y):
        db += -2*(y-(m*x+b))
        dm += -2*x*(y-(m*x+b))
        
    # Average the gradients
    db /= n
    dm /= n
    
    # Update new slope and y intercept
    b -= learning_rate * db
    m -= learning_rate * dm
    
    return m, b

def gradient_descent(x, y, m, b, epoch = 1000, learning_rate = 0.00001):
    """Descends / loops through to find lowest point

    Args:
        x (list / np arary): x values
        y (list / np arary): y values
        m (int): slope
        b (int): y intercept
        epoch (int, optional): Iterations. Defaults to 1000.
        learning_rate (float, optional): Learning_rate. Defaults to 0.00001.

    Returns:
        m, b: m = slope, b = y intercept
    """
    for i in range(epoch):
        m, b = gradient(x = x, y = y, m = m, b = b, learning_rate = learning_rate)
            
    return m, b
