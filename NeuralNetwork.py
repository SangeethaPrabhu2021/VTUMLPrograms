import tensorflow as tf
from tensorflow import keras

# Sample data
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

# Create a simple feedforward neural network
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100)

# Make predictions
predictions = model.predict([6.0])

print("Neural Network Predicts:", predictions[0][0])
