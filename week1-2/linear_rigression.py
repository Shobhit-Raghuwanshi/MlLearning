def predict(x, w, b):
    return w*x + b


def compute_cost(x , y , w, b):
    cost = 0
    for i in range(0 , len(x)):
        cost = cost + ((predict(x[i], w,b) - y[i])*(predict(x[i], w, b) - y[i]))

    
    return cost/(2*len(x))

def compute_gradient(x, y, w, b):
    gradiend_w = 0
    gradiend_b = 0 
    m = len(x)
    for i in range(0, len(x)):
        gradiend_w = gradiend_w + (predict(x[i], w,b) - y[i])*(x[i])
        gradiend_b = gradiend_b + (predict(x[i], w,b) - y[i])

    return gradiend_w/len(x) , gradiend_b/len(x)



def gradient_descent(x, y, w, b, alpha, iterations):
    # Repeat the following:
    # 1. Calculate gradients
    # 2. Update w and b
    # 3. Optionally track the cost

    for i in range(0 , iterations):
        dj_dw , dj_db = compute_gradient(x, y , w, b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        
    
    return w , b



crude_prices = [50, 60, 70, 80, 90]

w = 3
b = 10

for price in crude_prices:
    prediction = predict(price, w, b)
    print(f"Predicted value for crude price {price} is: {prediction}")


x_values = [50, 60, 70]
y_values = [120, 140, 160]

print(compute_cost(x_values, y_values, 3, 10))
print(compute_cost(x_values, y_values, 2, 20))

x = [50, 60, 70]
y = [120, 140, 160]

w = 3
b = 10

print(compute_gradient(x, y, w,b))

print("_____")


print(gradient_descent(x, y, w, b, 0.0001, 100))