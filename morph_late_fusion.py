import matplotlib.pyplot as plt
import numpy as np
import argparse
import keras
from collections import Counter
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Flatten, LeakyReLU, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D, Input
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
sys.path.insert(1, "/home/sasgari1/flatworms/utilities")
sys.path.insert(1, "/home/alehner1/summer20/flatworms/utilities")
from focal_loss import focal_loss
keras.losses.focal_loss_fixed = focal_loss()

import util
import metrics
from evaluation import analyze, first_image_predict, vote_predict, probability_predict


def assemble_mini_net(base_model, input_shape, input_layer, layer_num):
    """
    a = Conv2D(96, (11,11), strides=3, padding="same", input_shape=input_shape) (input_layer)
    a = BatchNormalization() (a)
    a = MaxPooling2D(pool_size=(2, 2)) (a)

    a = Conv2D(256, (5,5), padding="same", input_shape=input_shape) (a)
    a = BatchNormalization() (a)
    a = MaxPooling2D(pool_size=(2, 2)) (a)

    a = Conv2D(384, (3,3), padding="same", input_shape=input_shape) (a)
    a = BatchNormalization() (a)
    a = Conv2D(384, (3,3), padding="same", input_shape=input_shape) (a)
    a = BatchNormalization() (a)
    a = Conv2D(256, (3,3), padding="same", input_shape=input_shape) (a)
    a = BatchNormalization() (a)
    a = MaxPooling2D(pool_size=(2, 2)) (a)
    """


    #base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
    base_model.name = "Xception" + str(layer_num)
    for layer in base_model.layers:
        layer.name = layer.name + str(layer_num)

    output = base_model(input_layer)
    avg = keras.layers.GlobalAveragePooling2D()(output)

    return avg

def build_net(input_shape = (40, 128, 160, 3), num_types = 4, dropout = 0.5):
    input0 = keras.Input(shape=input_shape, name="input0")
    input1 = keras.Input(shape=input_shape, name="input1")
    input2 = keras.Input(shape=input_shape, name="input2")
    input3 = keras.Input(shape=input_shape, name="input3")

    base_model = keras.applications.Xception(weights="imagenet", include_top=False)
    a = assemble_mini_net(base_model, input_shape, input0, 0)
    b = assemble_mini_net(base_model, input_shape, input1, 1)
    c = assemble_mini_net(base_model, input_shape, input2, 2)
    d = assemble_mini_net(base_model, input_shape, input3, 3)

    fusion = keras.layers.concatenate([a, b, c, d])
    #fusion = Flatten() (fusion)
    """
    fusion = Dense(4096, activation="relu") (fusion)
    fusion = BatchNormalization() (fusion)
    fusion = Dropout(dropout) (fusion)
    fusion = Dense(4096, activation="relu") (fusion)
    fusion = BatchNormalization() (fusion)
    fusion = Dropout(dropout) (fusion)
    """

    output = Dense(num_types, activation="softmax") (fusion)

    neural_net = keras.Model(
        inputs = [input0, input1, input2, input3],
        outputs = [output]
    )
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    runmeta = tf.RunMetadata()

    neural_net.compile(optimizer="Nadam", loss = focal_loss(),
                       metrics=['accuracy', metrics.recall, metrics.precision], options = run_opts, run_metadata = runmeta)
    return neural_net

def test(model, X_test, y_test):
    loss, accuracy, recall, precision = model.evaluate({"input0": X_test[:,0], "input1": X_test[:,5], "input2": X_test[:,10], "input3": X_test[:,15]}, y_test)
    print("accuracy: {}%".format(accuracy*100))
    print("recall: {}%".format(recall*100))
    print("precision: {}%".format(precision*100))
    outputs3 = model.predict({"input0": X_test[:,0], "input1": X_test[:,5], "input2": X_test[:,10], "input3": X_test[:,15]})
    answers3 = [np.argmax(output) for output in outputs3]
    targets3 = [np.argmax(truth) for truth in y_test]
    util.analyze(answers3, targets3, [0, 1, 2, 3])


def filter_classes(X, y):
    shape = X.shape
    X_flat = np.reshape(X, (shape[0], shape[1] * shape[2] * shape[3]))
    y_flat = np.reshape(y, (shape[0], 1))

    combined_data = np.concatenate((X_flat, y_flat), axis = 1)

    combined_data_0 = combined_data[combined_data[:,-1] == 0] #normal
    combined_data_2 = combined_data[combined_data[:,-1] == 2] #contracted
    combined_data_2[:,-1] -= 1
    combined_data_3 = combined_data[combined_data[:,-1] == 3] #c-shape
    combined_data_3[:,-1] -= 1
    combined_data_4 = combined_data[combined_data[:,-1] == 4] #screw shaped
    combined_data_4[:,-1] -= 1

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

    #upsampling using smote
    shape = X_new_train.shape
    smote = SMOTE()
    X_new_train = X_new_train.reshape(shape[0], shape[1]*shape[2]*shape[3])

    X_new_train, y_new_train = smote.fit_resample(X_new_train, y_new_train)
    X_new_train = X_new_train.reshape(X_new_train.shape[0], shape[1], shape[2], shape[3], 1)
    X_new_train, y_new_train = shuffle(X_new_train, y_new_train)

    return X_new_train, y_new_train


def main():
    num_types = 4
    #hyperparameters:
    n_epochs = 100
    start_frame = 10
    jump_frame = 2
    n_class_samples = 20
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


    X_train = np.concatenate([X_train, X_train, X_train], axis=-1)
    X_val = np.concatenate([X_val, X_val, X_val], axis=-1)

    model = build_net((X_train.shape[2], X_train.shape[3], X_train.shape[4]), num_types, dropout)
    # using a model
    # MODEL HAS BEEN MOVED TO SONIDATA/FLATWORMS
    # model.load_weights('/sonidata/flatworms/curr_models/7_12_overnight_bigg_model_platewise.h5', by_name=True)
    # model = load_model('7_12_overnight_bigg_model_platewise.h5', custom_objects = {'recall': metrics.recall, 'precision': metrics.precision})

    history = model.fit({"input0": X_train[:,0], "input1":X_train[:,5], "input2":X_train[:,10], "input3":X_train[:,15]}, y_train,
                        validation_data=[{"input0": X_val[:,0], "input1": X_val[:,5], "input2": X_val[:,10], "input3": X_val[:,15]}, y_val],
                        epochs=n_epochs,
                        callbacks=[EarlyStopping(patience = 20)])

    X_train = None
    X_val = None
    X_test = np.concatenate([X_test, X_test, X_test], axis=-1)

    test(model, X_test, y_test)


if __name__ == '__main__':
    main()
