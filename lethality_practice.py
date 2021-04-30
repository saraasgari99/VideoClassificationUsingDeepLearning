import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import Counter

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from keras.regularizers import l1,l2
from numpy.random import randint
from sklearn.utils import shuffle, resample
from skimage import io

from imblearn.over_sampling import SMOTE
from imblearn.keras import BalancedBatchGenerator
import keras.losses

import sys
sys.path.insert(1, "/home/sasgari1/flatworms/utilities")
from focal_loss import focal_loss
keras.losses.focal_loss_fixed = focal_loss()

import util
import metrics

def build_net(input_shape):
    neural_net = Sequential()

    neural_net.add(Conv2D(32, (7,7), padding="same", input_shape=input_shape))
    neural_net.add(BatchNormalization())
    neural_net.add(MaxPooling2D(pool_size=(2, 2)))

    neural_net.add(Conv2D(64, (5,5), padding="same"))
    neural_net.add(BatchNormalization())
    neural_net.add(Conv2D(64, (5,5), padding="same"))
    neural_net.add(BatchNormalization())
    neural_net.add(MaxPooling2D(pool_size=(2, 2)))

    neural_net.add(Conv2D(128, (3,3), padding="same"))
    neural_net.add(BatchNormalization())
    neural_net.add(Conv2D(128, (3,3), padding="same"))
    neural_net.add(BatchNormalization())
    neural_net.add(MaxPooling2D(pool_size=(2, 2)))

    neural_net.add(Flatten())
    neural_net.add(Dense(100, activation="relu"))
    neural_net.add(BatchNormalization())
    neural_net.add(Dropout(0.5))

    neural_net.add(Dense(100, activation="relu"))
    neural_net.add(BatchNormalization())
    neural_net.add(Dropout(0.5))

    neural_net.add(Dense(1, activation="sigmoid"))

    neural_net.compile(optimizer="Nadam", loss = "binary_crossentropy",
                       metrics=['accuracy', metrics.recall, metrics.precision])
    return neural_net

def test(model, X_test, y_test):
    loss, accuracy, recall, precision = model.evaluate(X_test, y_test)
    print("accuracy: {}%".format(accuracy*100))
    outputs = model.predict(X_test)
    answers = [output[0] > 0.5 for output in outputs]
    print(answers)
    targets = [truth for truth in y_test]
    print(targets)
    util.analyze(answers, targets, [0,1])

def main():
    n_epochs = 100

    X_full = np.load('/sonidata/flatworms/dchaike1_scratch/dchaike1/flatworm/small_dead_X.npy')
    y_full = np.load('/sonidata/flatworms/dchaike1_scratch/dchaike1/flatworm/small_dead_y.npy')

    X_full = np.expand_dims(X_full, -1)

    X_full, y_full = shuffle(X_full, y_full)

    plt.imshow(X_full[0, :, :, 0], cmap="gray")
    plt.show()

    n = X_full.shape[0]
    train_ind = int(.8*n)
    val_ind = int(.9*n)

    X_train = X_full[:train_ind]
    y_train = y_full[:train_ind]
    X_val = X_full[train_ind:val_ind]
    y_val = y_full[train_ind:val_ind]
    X_test = X_full[val_ind:]
    y_test = y_full[val_ind:X_full.shape[0]]

    X_raw_train = X_train
    y_raw_train = y_train

    print(X_train.shape)
    print(y_train.shape)

    X_raw_train = np.reshape(X_raw_train, (X_raw_train.shape[0], X_raw_train.shape[1] * X_raw_train.shape[2]))
    y_raw_train = np.reshape(y_raw_train, (y_raw_train.shape[0], 1))
    print(X_raw_train.shape)
    print(y_raw_train.shape)

    combined_train = np.concatenate((X_raw_train, y_raw_train),axis=1)

    combined_train_0 = combined_train[combined_train[:,-1] == 0]
    combined_train_1 = combined_train[combined_train[:,-1] == 1]

    combined_train_0 = resample(combined_train_0,
                                replace=False,
                                n_samples=combined_train_1.shape[0],
                                random_state=123)
    combined_train = np.concatenate([combined_train_0, combined_train_1])

    new_y_train = combined_train[:, -1]
    new_X_train = combined_train[:, :-1]

    print(new_y_train.shape)
    print(new_X_train.shape)

    y_train = new_y_train
    X_train = np.reshape(new_X_train, (new_X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

    model = build_net((32, 40, 1))

    checkpoint_cb = ModelCheckpoint("my_model.h5", save_best_only = True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=n_epochs,
                        callbacks=[checkpoint_cb, EarlyStopping(patience=10)])

    model.load_weights("my_model.h5", by_name = True)

    test(model, X_test, y_test)


if __name__ == '__main__':
    main()
