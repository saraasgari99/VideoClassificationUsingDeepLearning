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
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
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

import metrics
from evaluation import analyze, first_image_predict, vote_predict, probability_predict, sum_confusion_matrix

def build_net(input_shape = (128, 160, 1), num_types = 4, dropout = 0.4, n_layer1 = 1, n_layer2 = 2, n_layer3 = 2, n_layer4 = 2):
    neural_net = Sequential()

    neural_net.add(Conv2D(32, (5,5), padding="same", input_shape=input_shape))
    neural_net.add(BatchNormalization())
    neural_net.add(Dropout(dropout))
    for i in range(n_layer1):
        neural_net.add(Conv2D(32, (5,5), padding="same", input_shape=input_shape))
        neural_net.add(BatchNormalization())

    neural_net.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(n_layer2):
        neural_net.add(Conv2D(64, (3,3), padding="same"))
        neural_net.add(BatchNormalization())

    neural_net.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(n_layer3):
        neural_net.add(Conv2D(128, (3,3), padding="same"))
        neural_net.add(BatchNormalization())

    neural_net.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(n_layer4):
        neural_net.add(Conv2D(256, (3,3), padding="same"))
        neural_net.add(BatchNormalization())

    neural_net.add(MaxPooling2D(pool_size=(2, 2)))
    neural_net.add(Dropout(dropout))

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

    neural_net.compile(optimizer= "Nadam", loss = focal_loss(),
                       metrics=['accuracy', metrics.recall, metrics.precision])
    return neural_net

def test(model, X_test, y_test):

    #getting the customized test sets (X only) for each testing method:
    first_image_pred = first_image_predict(model, X_test)
    probability_pred = probability_predict(model, X_test)
    vote_pred = vote_predict(model, X_test)

    print("\nusing first image:")
    analyze(first_image_pred, y_test)

    print("\nwith probabilities:")
    analyze(probability_pred, y_test)

    print("\nafter voting:")
    analyze(probability_pred, y_test)


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

    types_num = y_set.shape[1]
    frames_num = X_set.shape[1]
    frames_num = int((frames_num - start_frame)/jump_frame)
    X_new_set = np.zeros((X_set.shape[0]*frames_num, X_set.shape[2], X_set.shape[3], X_set.shape[4]))
    y_new_set = np.zeros((X_set.shape[0]*frames_num, types_num))

    num = 0
    for i, sequence in enumerate(X_set):
        for j in range(frames_num):
            # starting frame start_frame
            X_new_set[num] = sequence[start_frame + (j * jump_frame)]
            y_new_set[num] = y_set[i]
            num += 1

    return X_new_set, y_new_set

def balance_train(X_train, y_train, num_types, class_samples):

    X_raw_train = X_train
    y_raw_train = y_train
    X_raw_train = np.reshape(X_raw_train, (X_raw_train.shape[0], X_raw_train.shape[1] * X_raw_train.shape[2] * X_raw_train.shape[3]))
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
    X_new_train = np.reshape(X_raw_train, (X_raw_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]))

    y_train = None
    X_train = None
    y_raw_train = None
    X_raw_train = None
    combined_train = None
    combined_train_0 = None
    combined_train_rest = None
    combined_train_balanced = None

    X_new_train, y_new_train = shuffle(X_new_train, y_new_train)

    return X_new_train, y_new_train

def tune_params(param_grid, X_train, y_train, n_epochs = 15, n_folds = 3, n_iter = 5):

    # model class to use in the scikit random search CV
    model = KerasClassifier(build_fn=build_net, epochs=n_epochs, batch_size=20, verbose=1)

    grid = RandomizedSearchCV(estimator=model, cv=KFold(n_folds), param_distributions=param_grid,
                          verbose=20,  n_iter=n_iter, n_jobs=1)

    grid_result = grid.fit(X_train, y_train)
    best_params = grid_result.best_params_
    print(best_params)
    return best_params

def main():
    #hyperparameters:
    num_types = 4
    n_epochs = 100
    start_frame = 10
    jump_frame = 2
    class_samples = {0: 6000, 1: 6000, 2: 5000, 3: 2000}
    es_patience = 20
    dropout = 0.5
    n_splits = 5

    X = np.load('/sonidata/flatworms/dchaike1_scratch/dchaike1/new_shape_stack_X.npy')
    y = np.load('/sonidata/flatworms/dchaike1_scratch/dchaike1/new_shape_stack_y.npy')

    print("hello")

    #filtering some of the classes from the dataset
    X, y = filter_classes(X, y)

    X = np.expand_dims(X, -1)
    print('Original dataset shape %s' % Counter(y))
    print(X.shape)
    print(y.shape)

    X, y = shuffle(X, y)
    matrices = np.zeros((n_splits, num_types, num_types))
    total_test_pred = None
    total_test_ans = None
    #implementing cross-validation:
    skf = StratifiedKFold(n_splits = n_splits)
    i_fold = 0
    for rest_ind, test_ind in skf.split(X, y):
        X_test, y_test = X[test_ind], y[test_ind]
        X_rest, y_rest = X[rest_ind], y[rest_ind]
        #X_train, y_train = X_rest, y_rest

        val_skf = StratifiedKFold(n_splits = n_splits - 1)
        for train_ind, val_ind in skf.split(X_rest, y_rest):
            X_val, y_val = X_rest[val_ind], y_rest[val_ind]
            X_train, y_train = X_rest[train_ind], y_rest[train_ind]
            break
        print('split trainset shape %s' % Counter(y_train))
        print(X_train.shape)
        print(y_train.shape)
        #converting the labels to one-hot-vector
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        y_val = to_categorical(y_val)

        X_train, y_train = extract_frames(X_train, y_train, start_frame, jump_frame)

        print('extracted trainset shape %s' % Counter(y_train.argmax(axis=-1)))
        print(X_train.shape)
        print(y_train.shape)
        X_val, y_val = extract_frames(X_val, y_val, num_types, jump_frame)

        #starting from start_frame in the test set
        X_new_test = np.zeros((X_test.shape[0], X_test.shape[1]-start_frame, X_test.shape[2], X_test.shape[3], X_test.shape[4]))
        y_new_test = np.ones((X_test.shape[0], num_types))
        for i, sequence in enumerate(X_test):
            X_new_test[i] = sequence[start_frame:]
            y_new_test[i] = y_test[i]
        X_test, y_test = X_new_test, y_new_test

        #solving class imbalance in the training set
        print('balanced trainset shape %s' % Counter(y_train.argmax(axis=-1)))
        print(X_train.shape)

        base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
        avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = keras.layers.Dense(num_types, activation="softmax")(avg)
        model = keras.Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer= "Nadam", loss = focal_loss(),
                           metrics=['accuracy', metrics.recall, metrics.precision])

        X_train = np.concatenate([X_train, X_train, X_train], axis=-1)
        X_val = np.concatenate([X_val, X_val, X_val], axis=-1)
        history = model.fit(X_train, y_train,
                            validation_data=[X_val, y_val],
                            epochs=n_epochs,
                            callbacks=[EarlyStopping(patience=es_patience)])

        X_train = None
        X_val = None
        X_test = np.concatenate([X_test, X_test, X_test], axis=-1)

        print("fold " + str(i_fold))

        y_test = [np.argmax(truth) for truth in y_test]
        test(model, X_test, y_test)
        test_pred = probability_predict(model, X_test)
        test_ans = y_test

        if(i_fold == 0):
            total_test_pred = test_pred
            total_test_ans = test_ans
        else:
            total_test_pred = np.concatenate([total_test_pred, test_pred], axis = 0)
            total_test_ans = np.concatenate([total_test_ans, test_ans], axis = 0)

        test_pred = None
        test_ans = None

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

        save_command = input("Save this model? (yes/no)")
        if save_command == 'yes':
            model_name = input("Name your model: ")
            model.save(model_name + str(i_fold) + ".h5")
            print("The model is saved!")

        i_fold += 1

    print("Overall Performance using Probabilities Method")
    analyze(total_test_pred, total_test_ans)



if __name__ == '__main__':
    main()
