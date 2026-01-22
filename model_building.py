import tensorflow as tf
import keras
from keras import layers
from keras.optimizers import Adam

def build_model(input_dim):
   model = keras.Sequential()

   # Experimental: 128 Neuron
   model.add(layers.Dense(128, input_dim=input_dim, activation='relu'))
   model.add(layers.Dropout(0.4))

   # Layer 1: 64 Neuron
   model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))
   model.add(layers.Dropout(0.3))

   # Layer 2: 32 neuron
   model.add(layers.Dense(32, activation='relu'))
   # model.add(layers.Dropout(0.3))

   # Layer 3: 1 neuron
   model.add(layers.Dense(1, activation="sigmoid"))

   opt = Adam(learning_rate=0.0001)

   model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

   return model