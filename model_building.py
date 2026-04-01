import tensorflow as tf
import keras
from keras import layers
from keras.optimizers import Adam

def build_model(input_dim):
   """
   Construct a Deep Learning model (Multi-Layer Perceptron) for binary classification.

   The model consists of Dense layers with Dropout for regularization, using ReLU 
   activation for hidden layers and Sigmoid for the output layer.

   Args:
      input_dim (int): The number of input features.

   Returns:
      keras.Model: The compiled Keras model.
   """
   model = keras.Sequential()

   # Layer 1: 64 Neurons with ReLU activation
   model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))
   # Dropout to prevent overfitting
   model.add(layers.Dropout(0.3))

   # Layer 2: 32 Neurons with ReLU activation
   model.add(layers.Dense(32, activation='relu'))

   # Layer 3: Output layer with 1 neuron and Sigmoid activation
   # Ideal for binary classification tasks
   model.add(layers.Dense(1, activation="sigmoid"))

   # Adam optimizer with a low learning rate for stable training
   opt = Adam(learning_rate=0.0001)

   # Compile the model using binary crossentropy loss
   model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

   return model