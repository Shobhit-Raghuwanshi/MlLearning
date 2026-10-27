import numpy as np

# crude_prices = np.array([50, 60, 70, 80, 90])

# print(type(crude_prices))

# print(crude_prices*3 + 10)


def predict_multiple(x, w, b):
    return np.dot(x, w) + b


# print(predict_multiple( np.array([10,3,1]), np.array([-2,-1.5,4]), 5))

# print(np.array([10,3,1])*np.array([-2,-1.5,4]))



X = np.array([
    [10, 3, 1],
    [5, 2, 0],
    [-2, -1, -0.5]
])

y = np.array([-15, -10, 8])

w = np.array([-2, -1.5, 4])
b = 5

def predict_vectorized(X, w, b):
    
    return X @ w + b


print(predict_vectorized(X,w,b))



def compute_cost_vectorized(X, y, w, b):
    # Calculate predictions for all observations
    # Calculate the errors
    # Square the errors
    # Sum the squared errors
    # Divide by 2*m

    predictions = predict_vectorized(X, w, b)
    errs = predictions - y

    errs_square = errs*errs

    sum = np.sum(errs_square)

    return sum/2*np.shape(errs_square)[0]


# def compute_gradient(X, w, b)



print(compute_cost_vectorized(X,y,w,b))