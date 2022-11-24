import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix without Normalization')

    print(cm)

    thresh=cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j]>thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def train_classifier():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    if(len(physical_devices)>0) :
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    os.mkdir('../Classification_data')

    os.mkdir('../Classification_data/train')
    os.mkdir('../Classification_data/train/Healthy')
    os.mkdir('../Classification_data/train/Schizophrenic')

    os.mkdir('../Classification_data/valid')
    os.mkdir('../Classification_data/valid/Healthy')
    os.mkdir('../Classification_data/valid/Schizophrenic')

    os.mkdir('../Classification_data/test')
    os.mkdir('../Classification_data/test/Healthy')
    os.mkdir('../Classification_data/test/Schizophrenic')

    # folderPath='./CNN_data/'
    # destPath='./Classification_data/'

    for c in random.sample(glob.glob('../CNN_data/Healthy/*'), 161):
        shutil.copy(c, '../Classification_data/train/Healthy')

    for c in random.sample(glob.glob('../CNN_data/Schizophrenic/*'), 170):
        shutil.copy(c, '../Classification_data/train/Schizophrenic')

    for c in random.sample(glob.glob('../CNN_data/Healthy/*'), 50):
        shutil.copy(c, '../Classification_data/valid/Healthy')

    for c in random.sample(glob.glob('../CNN_data/Schizophrenic/*'), 50):
        shutil.copy(c, '../Classification_data/valid/Schizophrenic')

    for c in random.sample(glob.glob('../CNN_data/Healthy/*'), 50):
        shutil.copy(c, '../Classification_data/test/Healthy')

    for c in random.sample(glob.glob('../CNN_data/Schizophrenic/*'), 50):
        shutil.copy(c, '../Classification_data/test/Schizophrenic')

    train_path = '../Classification_data/train'
    valid_path = '../Classification_data/valid'
    test_path = '../Classification_data/test'

    train_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=['Healthy', 'Schizophrenic'], batch_size=10)

    valid_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=valid_path, target_size=(224,224), classes=['Healthy', 'Schizophrenic'], batch_size=10)

    test_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes=['Healthy', 'Schizophrenic'], batch_size=10, shuffle=False)

    model=Sequential([
        # By defining the input_shape we define the very first layer i.e. input layer.
        # The input_shape is the shape of the image that we are going to feed to the model.
        Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(224,224,3)),
        MaxPooling2D(pool_size=(2,2), strides=2),
        Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2), strides=2),
        Flatten(),
        Dense(units=2, activation='softmax')
    ])

    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    model.fit(
        x=train_batches,
        validation_data=valid_batches,
        epochs=10,
        verbose=2
    )

    predictions=model.predict(x=test_batches, verbose=0)
    predictions

    # Round of predictions
    np.round(predictions)   

    cm=confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

    test_batches.class_indices

    cm_plot_labels=['Healthy', 'Schizophrenic']
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

    # Calculate the accuracy
    acc=np.sum(np.diag(cm))/np.sum(cm)
    print('Accuracy: ', acc)

if __name__=='__main__':
    print('Classifying')