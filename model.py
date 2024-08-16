import os        
import numpy as np # linear algebra
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import PIL
import PIL.Image
from tensorflow import keras
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def train_test_gait():
    
    
    # Defines & compiles the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(rate=0.15), #adding dropout regularization throughout the model to deal with overfitting
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(rate=0.1),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(rate=0.10),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),

    # 3 output neuron for the 3 classes of Animal Images
    tf.keras.layers.Dense(3, activation='softmax')
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['acc'])
        

    # Creates an instance of an ImageDataGenerator called train_datagen, and a train_generator, train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    #splits data into training and testing(validation) sets
    train_datagen =ImageDataGenerator(rescale=1./255, validation_split=0.25)
    
    import matplotlib.pyplot as plt

   
    #training data
    train_generator = train_datagen.flow_from_directory(
        'C:/Users/Pranav Mahesh/Desktop/Pranav/Capstone/Model/Train',  # Source directory
        target_size=(150, 150),  # Resizes images
        batch_size=15,
        class_mode='categorical',subset = 'training')
    

    epochs = 15
    #Testing data
    validation_generator = train_datagen.flow_from_directory(
    'C:/Users/Pranav Mahesh/Desktop/Pranav/Capstone/Model/Train',
    target_size=(150, 150),
    batch_size=15,
    class_mode='categorical',
    subset='validation') # set as validation data
       
    #Model fitting for a number of epochs
    history = model.fit_generator(
      train_generator,
      steps_per_epoch=150,
      epochs=epochs,
      validation_data = validation_generator,
      validation_steps = 50,
      verbose=1)
    
        
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_acccuracy']
    

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    #This code is used to plot the training and validation accuracy
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
 
    # returns accuracy of training
    print("Training Accuracy:"), print(history.history['accuracy'][-1])
    print("Testing Accuracy:"), print (history.history['val_acccuracy'][-1])


train_test_gait()