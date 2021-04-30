import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import Counter

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Flatten, LeakyReLU, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
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

def build_net(input_shape = (5, 128, 60, 1), num_types = 4, dropout = 0.5):
    neural_net = Sequential()

    neural_net.add(Conv3D(32, (5,5,5), padding="same", input_shape=input_shape))
    neural_net.add(BatchNormalization())
    neural_net.add(Conv3D(32, (5,5,5), padding="same", input_shape=input_shape))
    neural_net.add(BatchNormalization())
    neural_net.add(MaxPooling3D(pool_size=(1, 2, 2)))

    neural_net.add(Conv3D(64, (3,3,3), padding="same", input_shape=input_shape))
    neural_net.add(BatchNormalization())
    neural_net.add(Conv3D(64, (3,3,3), padding="same", input_shape=input_shape))
    neural_net.add(BatchNormalization())
    neural_net.add(MaxPooling3D(pool_size=(1, 2, 2)))

    neural_net.add(Conv3D(128, (3,3,3), padding="same", input_shape=input_shape))
    neural_net.add(BatchNormalization())
    neural_net.add(Conv3D(128, (3,3,3), padding="same", input_shape=input_shape))
    neural_net.add(BatchNormalization())
    neural_net.add(MaxPooling3D(pool_size=(1, 2, 2)))

    neural_net.add(Conv3D(256, (3,3,3), padding="same", input_shape=input_shape))
    neural_net.add(BatchNormalization())
    neural_net.add(Conv3D(256, (3,3,3), padding="same", input_shape=input_shape))
    neural_net.add(BatchNormalization())
    neural_net.add(MaxPooling3D(pool_size=(1, 2, 2)))

    neural_net.add(Flatten())
    neural_net.add(Dense(100, activation="relu"))
    neural_net.add(BatchNormalization())
    neural_net.add(Dropout(dropout))

    neural_net.add(Dense(100, activation="relu"))
    neural_net.add(BatchNormalization())
    neural_net.add(Dropout(dropout))

    neural_net.add(Dense(50, activation="relu"))
    neural_net.add(BatchNormalization())
    neural_net.add(Dropout(dropout))

    neural_net.add(Dense(num_types, activation="softmax"))

    neural_net.compile(optimizer="Nadam", loss = focal_loss(),
                       metrics=['accuracy', metrics.recall, metrics.precision])
    return neural_net

def test(model, X_test, y_test):
    X_test = X_test[:,10:,:,:,:]

    X_prob_test = np.zeros((X_test.shape[0], 5, 128, 160, 1))
    X_vote_test = np.zeros((X_test.shape[0], 5, 128, 160, 1))
    X_old_test = np.zeros((X_test.shape[0], 5, 128, 160, 1))
    for i, sequence in enumerate(X_test):
        sequence = np.reshape(sequence, (8, 5, 128, 160, 1))

        probabilities = model.predict(sequence)
        #print("probs", probabilities)
        predictions = probabilities.argmax(axis=-1)
        #print("predictions", predictions)
        sum_prob = np.sum(probabilities, axis=0)
        #print("sum_prob", sum_prob)
        prob_prediction = sum_prob.argmax(axis=-1)
        #print("prob_prediction", prob_prediction)
        common_key = Counter(predictions).most_common(1)[0][0]
        X_old_test[i] = sequence[0] #first frame is used for testing
        done = 0
        done2 = 0
        for j, prediction in enumerate(predictions):
            if done == 1 and done2 == 1:
                break
            if prediction == prob_prediction:
                X_prob_test[i] = sequence[j] #inserting a desired image on prob_test
                done = 1
            if prediction == common_key:
                X_vote_test[i] = sequence[j] #inserting a desired image on vote_test
                done2 = 1

    first_image_pred = model.predict(X_old_test)
    probability_pred = model.predict(X_prob_test)
    vote_pred = model.predict(X_vote_test)

    print("using first image")
    analyze(first_image_pred, y_test)
    print("using probabilities")
    analyze(probability_pred, y_test)
    print("using voting")
    analyze(vote_pred, y_test)


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

def balance_train(X_train, y_train, num_types, class_samples):

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
                                n_samples=class_samples[0],
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
    smote = SMOTE(class_samples)
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
    n_class_samples = 2400
    ea_patience = 20
    dropout = 0.5
    class_samples = {0:2400, 1:2400, 2:2400, 3:2400}

    X = np.load('/sonidata/flatworms/dchaike1_scratch/dchaike1/shape_stack_X.npy')
    y = np.load('/sonidata/flatworms/dchaike1_scratch/dchaike1/shape_stack_y.npy')

    print("early fusion")

    #filtering some of the classes from the dataset
    X, y = filter_classes(X, y)

    X = np.expand_dims(X, -1)
    X, y = shuffle(X, y)
    print('Original dataset shape %s' % Counter(y))
    print(X.shape)
    print(y.shape)

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
    X_train, y_train = extract_frames(X_train, y_train, start_frame, jump_frame)
    print('extracted trainset shape %s' % Counter(y_train.argmax(axis = -1)))
    print(X_train.shape)
    print(y_train.shape)
    X_val, y_val = extract_frames(X_val, y_val, start_frame, jump_frame)

    #solving class imbalance in the training set
    X_train, y_train = balance_train(X_train, y_train, num_types, class_samples)
    print('balanced trainset shape %s' % Counter(y_train.argmax(axis = -1)))
    print(X_train.shape)
    print(y_train.shape)

    model = build_net((X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]), num_types, dropout)

    history = model.fit(X_train, y_train,
                        validation_data=[X_val, y_val],
                        epochs=n_epochs,
                        callbacks=[EarlyStopping(patience=ea_patience)])

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    test(model, X_test, y_test)


if __name__ == '__main__':
    main()
