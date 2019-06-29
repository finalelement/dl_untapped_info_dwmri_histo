from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras import backend as K

from keras.layers import Input, Dense, Dropout, merge, concatenate, Convolution3D, Flatten
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Add
from keras.wrappers.scikit_learn import KerasRegressor

from keras.optimizers import SGD, adam, nadam, Adagrad, RMSprop
from keras.regularizers import l1,l2
from keras.callbacks import EarlyStopping, CSVLogger

from deep_pnas_py_src.utils import split_X_8th_order

def build_nn():
    ## Takes input of 8th order SH DWMRI and outputs 8th order SH FOD
    model = Sequential()
    # Input layer with dimension 1 and hidden layer i with 128 neurons.
    model.add(Dense(45, input_shape=(45,)))
    model.add(Dense(400))
    # model.add(Dropout(0.4))
    model.add(Activation("relu"))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(66))
    model.add(Activation("relu"))
    # Hidden layer k with 64 neurons.
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.2))
    # Output Layer.
    model.add(Dense(66))
    # model.add(Activation("relu"))
    model.add(Dense(200))
    # model.add(Dropout(0.1))
    model.add(Dense(45))
    # Model is derived and compiled using mean square error as loss
    # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss='mse', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    return model


def build_nn2():
    ## Takes input of 8th order SH DWMRI and outputs 10th order SH FOD

    model = Sequential()
    # Input layer with dimension 1 and hidden layer i with 128 neurons.
    model.add(Dense(45, input_shape=(45,)))
    model.add(Dense(400))
    # model.add(Dropout(0.4))
    model.add(Activation("relu"))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(66))
    model.add(Activation("relu"))
    # Hidden layer k with 64 neurons.
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.2))
    # Output Layer.
    model.add(Dense(66))
    # model.add(Activation("relu"))
    model.add(Dense(200))
    # model.add(Dropout(0.1))
    model.add(Dense(66))
    # Model is derived and compiled using mean square error as loss
    # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss='mse', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    return model

def build_nn_resnet():
    # Batch Size 1000 and 250 epochs, hits a median of 0.8125
    input_dims = 45
    inputs = Input(shape=(input_dims,))

    #split0 = Input(shape=(1,))
    #split1 = Input(shape=(5,))
    #split2 = Input(shape=(9,))

    # Split 0 is 0th order, Split 1 is 2nd order, Split 2 is 4th order
    # split0, split1, split2 = tf.split(inputs, [1, 5, 9], 1)

    # 0th Order Network Flow
    x1 = Dense(400, activation='relu')(inputs)
    x2 = Dense(45, activation='relu')(x1)
    x3 = Dense(200, activation='relu')(x2)
    x4 = Dense(45,activation='linear')(x3)
    res_add = Add()([x2,x4])
    x5 = Dense(200,activation='relu')(res_add)
    x6 = Dense(45,activation='linear')(x5)

    model = Model(input=inputs, output=x6)

    opt_func = RMSprop(lr=0.0001)
    model.compile(loss='mse', optimizer=opt_func)
    print(model.summary())
    return model


def build_nn_resnet_v2():
    # For 250 epochs, batch size of 1000, median acc reached of 0.8015.
    input_dims = 45
    inputs = Input(shape=(input_dims,))

    #split0 = Input(shape=(1,))
    #split1 = Input(shape=(5,))
    #split2 = Input(shape=(9,))

    # Split 0 is 0th order, Split 1 is 2nd order, Split 2 is 4th order
    # split0, split1, split2 = tf.split(inputs, [1, 5, 9], 1)

    # 0th Order Network Flow
    x1 = Dense(400, activation='elu')(inputs)

    # First Res Block
    x2 = Dense(45, activation='elu')(x1)
    x3 = Dense(200, activation='elu')(x2)
    x4 = Dense(45,activation='elu')(x3)
    res_add = Add()([x2,x4])

    # Second Res Block
    x5 = Dense(45, activation='elu')(res_add)
    x6 = Dense(200, activation='elu')(x5)
    x7 = Dense(45, activation='elu')(x6)
    res_add_2 = Add()([x5, x7])

    # Third Res Block

    x8 = Dense(200,activation='elu')(res_add_2)
    x9 = Dense(45,activation='linear')(x8)

    model = Model(input=inputs, output=x9)

    opt_func = RMSprop(lr=0.0001)
    model.compile(loss='mse', optimizer=opt_func)
    print(model.summary())
    return model


def build_nn_resnet_v3():

    input_dims = 45
    inputs = Input(shape=(input_dims,))

    # split0 = Input(shape=(1,))
    # split1 = Input(shape=(5,))
    # split2 = Input(shape=(9,))

    # Split 0 is 0th order, Split 1 is 2nd order, Split 2 is 4th order
    # split0, split1, split2 = tf.split(inputs, [1, 5, 9], 1)

    # 0th Order Network Flow
    #x1 = Dense(400, activation='elu')(inputs)

    # First Res Block
    x2 = Dense(45, activation='elu')(inputs)
    x3 = Dense(100, activation='elu')(x2)
    x4 = Dense(45, activation='elu')(x3)
    res_add = Add()([x2, x4])

    # Second Res Block
    x5 = Dense(45, activation='elu')(res_add)
    x6 = Dense(100, activation='elu')(x5)
    x7 = Dense(45, activation='elu')(x6)
    res_add_2 = Add()([x5, x7])

    # Third Res Block
    x8 = Dense(45, activation='elu')(res_add_2)
    x9 = Dense(100, activation='elu')(x8)
    x10 = Dense(45, activation='elu')(x9)
    res_add_3 = Add()([x8, x10])

    x11 = Dense(100, activation='elu')(res_add_3)
    x12 = Dense(45, activation='linear')(x11)

    model = Model(input=inputs, output=x12)

    opt_func = RMSprop(lr=0.0001)
    model.compile(loss='mse', optimizer=opt_func)
    print(model.summary())
    return model


def reference_nn_resnet():
    # input_dims = 15
    # inputs = Input(shape=(input_dims,))

    split0 = Input(shape=(1,))
    split1 = Input(shape=(5,))
    split2 = Input(shape=(9,))
    split3 = Input(shape=(13,))
    split4 = Input(shape=(17,))

    # Split 0 is 0th order, Split 1 is 2nd order, Split 2 is 4th order
    # split0, split1, split2 = tf.split(inputs, [1, 5, 9], 1)

    # 0th Order Network Flow
    x1 = Dense(45, activation='relu')(split0)
    x1 = Dense(90, activation='relu')(x1)
    x1 = Dense(1, activation='linear')(x1)

    # 2nd Order Network Flow
    x2 = Dense(10, activation='relu')(split1)
    x2 = Dense(20, activation='relu')(x2)
    x2 = Dense(5, activation='linear')(x2)

    # 4th Order Network Flow
    x4 = Dense(18, activation='relu')(split2)
    x4 = Dense(36, activation='relu')(x4)
    x4 = Dense(9, activation='linear')(x4)

    # 4th Order Network Flow
    x6 = Dense(26, activation='relu')(split3)
    x6 = Dense(52, activation='relu')(x6)
    x6 = Dense(13, activation='linear')(x6)

    # 4th Order Network Flow
    x8 = Dense(34, activation='relu')(split4)
    x8 = Dense(68, activation='relu')(x8)
    x8 = Dense(17, activation='linear')(x8)

    merged = concatenate([x1, x2, x4, x6, x8])
    #merged = Dense(1, activation='linear')(merged)

    model = Model(input=[split0, split1, split2,split3, split4], output=merged)

    opt_func = RMSprop(lr=0.0001)
    model.compile(loss='mse', optimizer=opt_func)
    print(model.summary())
    return model

def train_network(model_t, X, y, num_epoch=10, batch=1000, val_split=0.1, save_path='none'):

    model_t.fit(X, y, epochs=num_epoch, batch_size=batch, verbose=1, shuffle=True, validation_split=val_split)
    model_t.save_weights(save_path)
    return model_t

def train_refnet_network(model_t, X, y, num_epoch=10, batch=1000, val_split=0.05):

    x1,x2,x3,x4,x5 = split_X_8th_order(X)
    model_t.fit([x1,x2,x3,x4,x5], y, epochs=num_epoch, batch_size=batch, verbose=1, shuffle=True, validation_split=val_split)
    return model_t