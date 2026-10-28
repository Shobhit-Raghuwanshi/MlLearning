import numpy as np

# crude_prices = np.array([50, 60, 70, 80, 90])

# print(type(crude_prices))

# print(crude_prices*3 + 10)


def predict_multiple(x, w, b):
    return np.dot(x, w) + b


# print(predict_multiple( np.array([10,3,1]), np.array([-2,-1.5,4]), 5))

# print(np.array([10,3,1])*np.array([-2,-1.5,4]))



X = np.array([
    [1, 1000],
    [2, 2000],
    [3, 3000],
    [4, 4000]
], dtype=float)

y = np.array([2, 4, 6, 8], dtype=float)


w = np.array([2,3])
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

    return sum/(2*np.shape(errs_square)[0])


def compute_gradient(X, y , w, b):

    dj_dw = X.T @ (predict_vectorized(X, w,b) - y)/np.shape(X)[0]
    dj_db = np.sum(predict_vectorized(X, w,b)- y)/np.shape(X)[0]

    return dj_dw , dj_db


def gradient_descent(X, y, w, b, alpha, iterations):
    
   
    for i in range(iterations):
        dj_dw , dj_db = compute_gradient(X, y, w, b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

    return w, b


print("Starting w:", w)
print("Starting b:", b)
print("Starting cost:", compute_cost_vectorized(X, y, w, b))
    


def standardize_features(X):
    # Step 1: Har column ka mean nikalo

    # Step 2: Har column ka standard deviation nikalo

    # Step 3: Z-score formula apply karo

    # Step 4: Standardized matrix return karo
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean)/std


print(" X:")
print(X)

X_scaled = standardize_features(X)

print("Scaled X:")
print(X_scaled)

print("Mean:", np.mean(X_scaled, axis=0))
print("Std:", np.std(X_scaled, axis=0))
print("Shape:", X_scaled.shape)



alpha = 0.001
iterations = 100000

initial_cost = compute_cost_vectorized(X_scaled, y, w, b)

final_w, final_b = gradient_descent(
    X_scaled, y, w, b, alpha, iterations
)

final_cost = compute_cost_vectorized(
    X_scaled, y, final_w, final_b
)

print("Initial cost:", initial_cost)
print("Final cost:", final_cost)
print("Learned weights:", final_w)
print("Learned bias:", final_b)