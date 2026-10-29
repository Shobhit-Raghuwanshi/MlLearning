
import numpy as np

from sklearn.model_selection import train_test_split

X = np.array([
    [10, 5],
    [20, 8],
    [30, 10],
    [40, 14],
    [50, 17],
    [60, 20],
    [70, 23],
    [80, 27]
], dtype=float)

y = np.array(
    [80, 130, 170, 230, 280, 330, 380, 440],
    dtype=float
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



def fit_scaler(X_train):
    # Training data se mean aur std nikalo
    # Dono return karo
    return np.mean(X_train, axis=0), np.std(X_train, axis=0)


def transform_features(X, mean, std):
    # Given mean/std se X ko standardize karo
    # Naya mean/std calculate mat karna
    return (X - mean)/ std

X_train_mean , X_train_std = fit_scaler(X_train= X_train)
X_train_scaled = transform_features(X_train , X_train_mean , X_train_std)
X_test_scaled = transform_features(X_test, X_train_mean , X_train_std)