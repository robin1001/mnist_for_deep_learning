'''
Created on 2017-11-29
Author: Binbin Zhang
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from numpy import linalg as la

import numpy as np
import sys

batch_size = 128
num_classes = 10
epochs = 20
load_pretrained_model = True

def error_msg(msg):
    print(msg)
    sys.exit(-1)

def svd(m, k): 
    assert(len(m.shape) == 2)
    assert(k < m.shape[1])
    U, sigma, VT = la.svd(m)
    V = np.dot(np.diag(sigma[0:k]), VT[0:k, :])
    return U[:, 0:k], V

def apply_svd_to_model(model, k):
    svd_model = Sequential()
    cur = 0
    for layer in model.layers:
        layer_name = layer.name
        class_name = layer.__class__.__name__
        in_dim, out_dim = layer.input_shape[1], layer.output_shape[1]
        if class_name == 'Dense': 
            w = layer.kernel.get_value()
            b = layer.bias.get_value()
            if layer.activation.__name__ != 'softmax':
                U, V = svd(w, k)
                dense1 = Dense(k, input_shape=(in_dim,), use_bias=False, 
                    weights=((U,)), name=('svd_dense_%d_0' % cur))
                dense2 = Dense(out_dim, input_shape=(k,), use_bias=True, 
                    weights=((V,b)), name=('svd_dense_%d_1' % cur)) 
                svd_model.add(dense1)
                svd_model.add(dense2)
                #print(w.shape, U.shape, V.shape)
            else:
                dense = Dense(out_dim, input_shape=(in_dim,), activation='softmax',
                    use_bias=True, weights=((w,b)), name=('dense_%d' % cur))
                svd_model.add(dense)
            cur += 1
        elif class_name == 'Activation':
            svd_model.add(Activation(layer.activation.__name__))
        else:
            error_msg('error, layer %s %s is supported' % (layer_name, class_name))
    return svd_model

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if not load_pretrained_model:
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    model.save('dnn.model')
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
else:
    model = load_model('dnn.model')
    print('Raw model')
    num_raw_params = model.count_params()
    print('Number parameters:', num_raw_params)

for k in (256, 128, 64, 32, 16, 8):
    svd_model = apply_svd_to_model(model, k)
    svd_model.save('svd_%d.model' % k)
    #svd_model.summary()
    svd_model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    score = svd_model.evaluate(x_test, y_test)

    history = svd_model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=12,
                        validation_data=(x_test, y_test))
    svd_model.save('svd_%d_retrain.model' % k)
    retrain_score = svd_model.evaluate(x_test, y_test)

    print()
    print('SVD dim:', k)
    print(('Number parameters: %d compression ratio: %f' % (
           svd_model.count_params(), svd_model.count_params() / num_raw_params)))
    print('SVD test accuracy: %.4f, SVD retrain accuracy: %.4f' % (score[1], retrain_score[1]))

