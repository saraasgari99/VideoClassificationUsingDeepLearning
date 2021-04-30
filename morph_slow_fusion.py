import matplotlib.pyplot as plt
import numpy as np
import argparse
import keras
from collections import Counter

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Flatten, LeakyReLU, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D, Input, Lambda
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from keras.regularizers import l1,l2
from numpy.random import randint
from sklearn.utils import shuffle, resample
from skimage import io
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer

from imblearn.over_sampling import SMOTE
from imblearn.keras import BalancedBatchGenerator
import keras.losses

import sys
sys.path.insert(1, "/home/alehner1/summer20/flatworms/utilities")
from focal_loss import focal_loss
keras.losses.focal_loss_fixed = focal_loss()

import util
import metrics
from evaluation import analyze, first_image_predict, vote_predict, probability_predict

def assemble_mini_net(input_shape, input_layer, num_filters, kernel_size, num_strides, pool_size):
    a = Conv3D(num_filters, kernel_size, strides=num_strides, padding="same", input_shape=input_shape) (input_layer)
    a = BatchNormalization() (a)
    a = MaxPooling3D(pool_size=pool_size) (a)

    """
    a = Conv2D(32, (5,5), padding="same", input_shape=input_shape) (input_layer)
    a = BatchNormalization() (a)
    a = Conv2D(32, (5,5), padding="same", input_shape=input_shape) (a)
    a = BatchNormalization() (a)
    a = MaxPooling2D(pool_size=(2, 2)) (a)

    a = Conv2D(64, (3,3), padding="same", input_shape=input_shape) (a)
    a = BatchNormalization() (a)
    a = Conv2D(64, (3,3), padding="same", input_shape=input_shape) (a)
    a = BatchNormalization() (a)
    a = MaxPooling2D(pool_size=(2, 2)) (a)

    a = Conv2D(128, (3,3), padding="same", input_shape=input_shape) (a)
    a = BatchNormalization() (a)
    a = Conv2D(128, (3,3), padding="same", input_shape=input_shape) (a)
    a = BatchNormalization() (a)
    a = MaxPooling2D(pool_size=(2, 2)) (a)

    a = Conv2D(256, (3,3), padding="same", input_shape=input_shape) (a)
    a = BatchNormalization() (a)
    a = Conv2D(256, (3,3), padding="same", input_shape=input_shape) (a)
    a = BatchNormalization() (a)
    a = MaxPooling2D(pool_size=(2, 2)) (a)
    """

    return a

def build_net(input_shape = (40, 128, 60, 1), num_types = 4, dropout = 0.5):
    input = keras.Input(shape=input_shape)

    a0 = Lambda(lambda x: x[:,0:10]) (input)
    a0 = assemble_mini_net(input_shape, a0, 32, (10, 11, 11), 3, (1, 1, 1))
    a0 = assemble_mini_net(input_shape, a0, 64, (7, 7, 7), 3, (1, 2, 2))

    a1 = Lambda(lambda x: x[:,10:20]) (input)
    a1 = assemble_mini_net(input_shape, a1, 32, (10, 11, 11), 3, (1, 1, 1))
    a1 = assemble_mini_net(input_shape, a1, 64, (7, 7, 7), 3, (1, 2, 2))

    a2 = Lambda(lambda x: x[:,20:30]) (input)
    a2 = assemble_mini_net(input_shape, a2, 32, (10, 11, 11), 3, (1, 1, 1))
    a2 = assemble_mini_net(input_shape, a2, 64, (7, 7, 7), 3, (1, 2, 2))

    a3 = Lambda(lambda x: x[:,30:40]) (input)
    a3 = assemble_mini_net(input_shape, a3, 32, (10, 11, 11), 3, (1, 1, 1))
    a3 = assemble_mini_net(input_shape, a3, 64, (7, 7, 7), 3, (1, 2, 2))

    b0 = keras.layers.concatenate([a0, a1])
    b0 = assemble_mini_net(input_shape, b0, 64, (5, 5, 5), 1, (1, 1, 1))
    b0 = assemble_mini_net(input_shape, b0, 128, (5, 5, 5), 1, (1, 2, 2))

    b1 = keras.layers.concatenate([a2, a3])
    b1 = assemble_mini_net(input_shape, b1, 64, (5, 5, 5), 1, (1, 1, 1))
    b1 = assemble_mini_net(input_shape, b1, 128, (5, 5, 5), 1, (1, 2, 2))

    c0 = keras.layers.concatenate([b0, b1])
    c0 = assemble_mini_net(input_shape, c0, 256, (3, 3, 3), 1, (1, 1, 1))
    c0 = assemble_mini_net(input_shape, c0, 256, (3, 3, 3), 1, (2, 2, 2))

    fusion = Flatten() (c0)

    """
    fusion = Dense(1024, activation="relu") (fusion)
    fusion = BatchNormalization() (fusion)
    fusion = Dropout(dropout) (fusion)

    fusion = Dense(256, activation="relu") (fusion)
    fusion = BatchNormalization() (fusion)
    fusion = Dropout(dropout) (fusion)

    fusion = Dense(64, activation="relu") (fusion)
    fusion = BatchNormalization() (fusion)
    fusion = Dropout(dropout) (fusion)
    """

    fusion = Dense(512, activation="relu") (fusion)
    fusion = BatchNormalization() (fusion)
    fusion = Dropout(dropout) (fusion)
    fusion = Dense(256, activation="relu") (fusion)
    fusion = BatchNormalization() (fusion)
    fusion = Dropout(dropout) (fusion)
    fusion = Dense(64, activation="relu") (fusion)
    fusion = BatchNormalization() (fusion)
    fusion = Dropout(dropout) (fusion)

    output = Dense(num_types, activation="softmax") (fusion)

    neural_net = keras.Model(
        inputs = [input],
        outputs = [output]
    )

    neural_net.compile(optimizer="Nadam", loss = focal_loss(),
                       metrics=['accuracy', metrics.recall, metrics.precision])
    return neural_net

def test(model, X_test, y_test):
    loss, accuracy, recall, precision = model.evaluate(X_test, y_test)
    print("accuracy: {}%".format(accuracy*100))
    print("recall: {}%".format(recall*100))
    print("precision: {}%".format(precision*100))
    outputs3 = model.predict(X_test)
    answers3 = [np.argmax(output) for output in outputs3]
    targets3 = [np.argmax(truth) for truth in y_test]
    util.analyze(answers3, targets3, [0, 1, 2, 3])


def filter_classes(X, y):
    shape = X.shape
    X_flat = np.reshape(X, (shape[0], shape[1] * shape[2] * shape[3]))
    y_flat = np.reshape(y, (shape[0], 1))

    combined_data = np.concatenate((X_flat, y_flat), axis = 1)

    combined_data_0 = combined_data[combined_data[:,-1] == 0] #normal
    print(combined_data_0.shape[0])
    combined_data_2 = combined_data[combined_data[:,-1] == 2] #contracted
    combined_data_2[:,-1] -= 1
    print(combined_data_2.shape[0])
    combined_data_3 = combined_data[combined_data[:,-1] == 3] #c-shape
    combined_data_3[:,-1] -= 1
    print(combined_data_3.shape[0])
    combined_data_4 = combined_data[combined_data[:,-1] == 4] #screw shaped
    combined_data_4[:,-1] -= 1
    print(combined_data_4.shape[0])

    X_flat = None
    y_flat = None
    combined_data = None

    combined_data = np.concatenate((combined_data_0, combined_data_2, combined_data_3, combined_data_4))

    combined_data_0 = None
    combined_data_2 = None
    combined_data_3 = None
    combined_data_4 = None

    X_filtered = combined_data[:, :-1]
    X_filtered = np.reshape(X_filtered, (X_filtered.shape[0], shape[1], shape[2], shape[3]))
    y_filtered = combined_data[:, -1]

    return X_filtered, y_filtered

def extract_frames(X_set, y_set, start_frame, jump_frame):
    #X_set.shape:  (1470, 50, 128, 160, 1)
    #y_set.shape:  (1470, 4)
    types_num = y_set.shape[1]
    frames_num = 8
    X_new_set = np.zeros((X_set.shape[0]*frames_num, 5, X_set.shape[2], X_set.shape[3], X_set.shape[4]))
    y_new_set = np.zeros((X_set.shape[0]*frames_num, types_num))

    num = 0
    for i, sequence in enumerate(X_set):
        for j in range(frames_num):
            # starting frame start_frame
            X_new_set[num] = sequence[(start_frame + j * 5):(start_frame + j * 5) + 5]
            y_new_set[num] = y_set[i]
            num += 1

    return X_new_set, y_new_set

def balance_train(X_train, y_train, num_types, n_class_samples):

    X_raw_train = X_train
    y_raw_train = y_train
    X_raw_train = np.reshape(X_raw_train, (X_raw_train.shape[0], X_raw_train.shape[1] * X_raw_train.shape[2] * X_raw_train.shape[3] * X_raw_train.shape[4]))
    y_raw_train = np.reshape(y_raw_train, (y_raw_train.shape[0], num_types))
    combined_train = np.concatenate((X_raw_train, y_raw_train), axis=1)

    #separating the first class
    combined_train_0 = combined_train[combined_train[:,-num_types] == 1]
    combined_train_rest = combined_train[combined_train[:,-num_types] != 1]

    #downsampling the first class
    combined_train_0 = resample(combined_train_0,
                                replace=False,
                                n_samples=n_class_samples,
                                random_state=123)
    combined_train_balanced = np.concatenate([combined_train_0, combined_train_rest])
    combined_train_balanced = shuffle(combined_train_balanced)

    y_raw_train = combined_train_balanced[:, -num_types:]
    X_raw_train = combined_train_balanced[:, :-num_types]

    y_new_train = y_raw_train
    X_new_train = np.reshape(X_raw_train, (X_raw_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))

    y_train = None
    X_train = None
    y_raw_train = None
    X_raw_train = None
    combined_train = None
    combined_train_0 = None
    combined_train_rest = None
    combined_train_balanced = None

    """
    #upsampling using smote
    shape = X_new_train.shape
    smote = SMOTE()
    X_new_train = X_new_train.reshape(shape[0], shape[1]*shape[2]*shape[3])

    X_new_train, y_new_train = smote.fit_resample(X_new_train, y_new_train)
    X_new_train = X_new_train.reshape(X_new_train.shape[0], shape[1], shape[2], shape[3], 1)
    """

    X_new_train, y_new_train = shuffle(X_new_train, y_new_train)

    return X_new_train, y_new_train


def main():
    num_types = 4
    #hyperparameters:
    n_epochs = 200
    start_frame = 10
    jump_frame = 2
    n_class_samples = 300
    ea_patience = 20
    dropout = 0.5

    X = np.load('/sonidata/flatworms/dchaike1_scratch/dchaike1/new_shape_stack_X.npy')
    y = np.load('/sonidata/flatworms/dchaike1_scratch/dchaike1/new_shape_stack_y.npy')

    print("hello")

    #filtering some of the classes from the dataset
    X, y = filter_classes(X, y)

    X = np.expand_dims(X, -1)
    X, y = shuffle(X, y)

    X = X[:,10:]

    #implementing cross-validation
    skf = StratifiedKFold(n_splits = 5)
    for train_ind, test_ind in skf.split(X, y):
        X_test, y_test = X[test_ind], y[test_ind]
        X_rest, y_rest = X[train_ind], y[train_ind]

    X = None
    y = None

    skf = StratifiedKFold(n_splits = 4)
    for train_ind, test_ind in skf.split(X_rest, y_rest):
        X_val, y_val = X_rest[test_ind], y_rest[test_ind]
        X_train, y_train = X_rest[train_ind], y_rest[train_ind]

    X_rest = None
    y_rest = None

    #converting the labels to one-hot-vector
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)

    #extracting the frames in each sequence, starting from start_frame and jumping with jump_frame step

    #solving class imbalance in the training set
    X_train, y_train = balance_train(X_train, y_train, num_types, n_class_samples)

    model = build_net((X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]), num_types, dropout)

    # using a model
    # MODEL HAS BEEN MOVED TO SONIDATA/FLATWORMS
    # model.load_weights('/sonidata/flatworms/curr_models/7_12_overnight_bigg_model_platewise.h5', by_name=True)
    # model = load_model('7_12_overnight_bigg_model_platewise.h5', custom_objects = {'recall': metrics.recall, 'precision': metrics.precision})

    checkpoint_cb = ModelCheckpoint("practice_shape_nn_testing.h5", save_best_only = False)

    history = model.fit(X_train, y_train,
                        validation_data=[X_val, y_val],
                        epochs=n_epochs,
                        callbacks=[EarlyStopping(patience=30)])

    test(model, X_test, y_test)


if __name__ == '__main__':
    main()
