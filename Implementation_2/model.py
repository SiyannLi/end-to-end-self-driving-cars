import tensorflow as tf
from tensorflow.keras import layers, models

# Define the model using Keras Sequential API
def create_model():
    model = models.Sequential()

    # Input layer
    model.add(layers.InputLayer(input_shape=(66, 200, 3)))

    # First convolutional layer
    model.add(layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', padding='valid',
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=tf.keras.initializers.Constant(0.1)))

    # Second convolutional layer
    model.add(layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu', padding='valid',
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=tf.keras.initializers.Constant(0.1)))

    # Third convolutional layer
    model.add(layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu', padding='valid',
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=tf.keras.initializers.Constant(0.1)))

    # Fourth convolutional layer
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid',
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=tf.keras.initializers.Constant(0.1)))

    # Fifth convolutional layer
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid',
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=tf.keras.initializers.Constant(0.1)))

    # Flatten the output from conv layers
    model.add(layers.Flatten())

    # First fully connected layer
    model.add(layers.Dense(1164, activation='relu',
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=tf.keras.initializers.Constant(0.1)))

    # Dropout layer
    model.add(layers.Dropout(0.5))

    # Second fully connected layer
    model.add(layers.Dense(100, activation='relu',
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=tf.keras.initializers.Constant(0.1)))

    # Dropout layer
    model.add(layers.Dropout(0.5))

    # Third fully connected layer
    model.add(layers.Dense(50, activation='relu',
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=tf.keras.initializers.Constant(0.1)))

    # Dropout layer
    model.add(layers.Dropout(0.5))

    # Fourth fully connected layer
    model.add(layers.Dense(10, activation='relu',
                            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                            bias_initializer=tf.keras.initializers.Constant(0.1)))

    # Dropout layer
    model.add(layers.Dropout(0.5))

    # Output layer (using atan for steering angle scaling)
    model.add(layers.Dense(1))
    model.add(layers.Lambda(lambda x: tf.multiply(tf.atan(x), 2)))

    return model

# Create the model
model = create_model()

# Compile the model with optimizer and loss function
model.compile(optimizer='adam', loss='mse')

# Model summary
model.summary()

# Now you can train the model using model.fit() or perform inference using model.predict()
