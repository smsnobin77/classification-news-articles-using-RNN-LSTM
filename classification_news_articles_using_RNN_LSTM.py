import tensorflow as tf
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from tensorflow import keras
from tensorflow.keras import layers


import numpy as np
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = reuters.load_data(num_words=1000, test_split=0.2)
category = np.max(Y_train) + 1

print('category', category)
print('news articles to train', len(X_train))
print('news articles to test', len(X_test))
print(X_train[0])


#processing
x_train = pad_sequences(X_train, maxlen=100)
x_test = pad_sequences(X_test, maxlen=100)
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

model = Sequential()
model.add(Embedding(1000, 100))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(46, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=100, epochs=20, validation_data=(x_test, y_test))
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))