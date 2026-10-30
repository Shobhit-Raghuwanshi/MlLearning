import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

# Features: [interest-rate change, crude-price change]
X = np.array([
    [ 0.5,  2],
    [-0.3, -4],
    [ 0.8,  5],
    [-0.5, -2],
    [ 0.2,  3],
    [-0.7, -5],
    [ 0.6,  4],
    [-0.1, -1],
    [ 0.4,  1],
    [-0.4, -3],
    [ 0.9,  6],
    [-0.6, -4]
])

# Target: market return
y = np.array([
    -1.2, 2.1, -2.0, 1.4, -0.8, 2.6,
    -1.5, 0.5, -0.6, 1.8, -2.4, 2.2
])

X_train ,X_test ,y_train ,y_test = train_test_split(X, y , train_size=0.75 ,random_state=41, shuffle=True)


def features(X_train):
    return np.mean(X_train, axis=0) , np.std(X_train, axis=0)

def standardize_features(X , mean, std):
    return (X-mean)/std

X_train_mean , X_train_std =  features(X_train)

X_train_standardize = standardize_features(X_train, X_train_mean, X_train_std)
X_test_standardize = standardize_features(X_test, X_train_mean, X_train_std)

def cost_function(X, y , w, b):
    return np.sum((np.dot(X, w) + b - y)**2)/(2*np.shape(X)[0])

def predict(X, w, b):
    return np.dot(X,w) + b

def calculate_gradient(X,y,w,b):

    dj_dw = (X.T @ (predict(X,w,b) - y))/np.shape(X)[0]
    dj_db = np.sum(predict(X, w, b) - y)/np.shape(X)[0]

    return dj_dw , dj_db

def gradient_decend(X, y , w, b, alpha, iterations):
    for _ in range(iterations):
        dj_dw , dj_db = calculate_gradient(X, y, w, b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

    return w, b

def calculate_mae(y_actual, y_pred):
    return np.mean(np.abs(y_pred - y_actual))

def calculate_mse(y_actual, y_pred):
    return np.sum((y_pred - y_actual)**2)/np.shape(y_actual)[0]


w = np.zeros(np.shape(X_train_standardize)[1])
b = 0

print(w)

alpha = 0.001
iterations = 1000000

initial_cost = cost_function(X_train_standardize, y_train, w, b)

w , b = gradient_decend(X_train_standardize , y_train , w, b , alpha, iterations)

final_cost = cost_function(X_train_standardize, y_train , w, b)

print(w, b)

print("INITIAL_COST -->" , initial_cost)
print("FINAL_COST -->" , final_cost)

print("MAE" , calculate_mae(y_test, predict(X_test_standardize, w, b)))
print("MSE" , calculate_mse(y_test, predict(X_test_standardize, w, b)))


regr = LinearRegression()

regr.fit(X_train_standardize, y_train)

print(regr.intercept_, regr.coef_)

print(cost_function(X_train_standardize , y_train ,  regr.coef_ , regr.intercept_))

print("MAE" , calculate_mae(y_test, predict(X_test_standardize, regr.coef_, regr.intercept_)))
print("MSE" , calculate_mse(y_test, predict(X_test_standardize, regr.coef_, regr.intercept_)))

