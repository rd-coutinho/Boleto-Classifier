# -*- coding: utf-8 -*-
"""
@author: renat
"""
    
# Randomly choosing images to build the database
from os import walk
from random import seed
from random import choice
from random import sample
from random import shuffle
from PIL import Image
#Image.MAX_IMAGE_PIXELS = 100000000000 
Image.MAX_IMAGE_PIXELS = None

mypath = "C:\\Users\\renat\\OneDrive\\Área de Trabalho\\Classifier Boletos\\dataset population\\training_set\\Não Boletos (Population)"

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break

seed(1)

# Creating a random subset folder for training and test
shuffle(f)
trainingSet = f[0:12000]
testSet = f[12000:len(f)+1]

for i in range (1, len(testSet) + 1):
    imagem_teste = image.load_img('C:\\Users\\renat\\OneDrive\\Área de Trabalho\\Classifier Boleto\\dataset population\\training_set\\Não Boletos (Population)\\{}'.format(testSet[i-1]))
    #imagem_teste = image.img_to_array(imagem_teste)
    #imagem_teste /= 255
    imagem_teste= image.save_img('C:\\Users\\renat\\OneDrive\\Área de Trabalho\\Classifier Boleto\\dataset population\\training_set',
                   testSet[i-1])


##################################################################################################
# Developing the classifier
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from IPython.display import display, Image
import numpy as np
from keras.preprocessing import image
import time

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000 

data_path = "C:\\Users\\renat\\OneDrive\\Área de Trabalho\\Classifier Boletos"

classifier = Sequential()
classifier.add(Conv2D(16, (3, 3), input_shape = (192, 136, 1), activation = "relu"))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Dropout(0.1))
classifier.add(Conv2D(128, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Dropout(0.1))
classifier.add(Conv2D(128, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Dropout(0.1))
classifier.add(Conv2D(64, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Dropout(0.1))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = "relu"))
#classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, activation = "sigmoid"))  # Or softmax
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"]) # Or categorical_crossentropy

checkpointer = ModelCheckpoint(filepath=data_path + "\\classifier-{epoch:02d}.hdf5", verbose=1)

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
'''train_datagen = ImageDataGenerator(rescale=1./255)'''

'''train_datagen = ImageDataGenerator(rescale=1./255,                                
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True)'''

train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode="nearest",
                                   brightness_range=[0.2,1.0],
                                   
                                   rotation_range=90,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow(trainingSet,
                                      target_size=(192, 136),
                                      batch_size=32,
                                      class_mode='binary',
                                      color_mode='grayscale')

test_set = test_datagen.flow_from_directory('dataset aleatório/test_set',
                                            target_size=(192, 136),
                                            batch_size=32,
                                            class_mode='binary',
                                            color_mode='grayscale')

class_weight = {
        }

start = time.time()
# Code to start the training step
classifier.fit_generator(training_set,
                         steps_per_epoch=578,    
                         epochs=20,
                         validation_data=test_set,
                         validation_steps=144,
                         callbacks=[checkpointer],
                         class_weight=class_weight)  

classifier.save(data_path + "final_classifier.hdf5")

end = time.time()
print("Total time of processing: {} minutes".format((end-start)/60)) 


classifier.load_weights(data_path + "\\classifier-10.hdf5")


##################################################################################################
# Loop to see the validation accuracy (boletos) ################################
from os import walk
import time

mypath = "C:\\Users\\renat\\OneDrive\\Área de Trabalho\\Classifier Boletos\\dataset\\validation_set\\Boletos"

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break

resultados = []
for i in range (1, len(f) + 1):
    imagem_teste = image.load_img('C:\\Users\\renat\\OneDrive\\Área de Trabalho\\Classifier Boletos\\dataset\\validation_set\\Boletos\\{}'.format(f[i-1]),
                                  target_size = (192, 136),
                                  color_mode = 'grayscale')
    imagem_teste = image.img_to_array(imagem_teste)
    imagem_teste /= 255
    imagem_teste = np.expand_dims(imagem_teste, axis = 0)
    previsão = classifier.predict(imagem_teste)
    resultados.append(previsão)
    
erros = 0 
n = 0
for i in resultados:
    if float(i[0][0]) > 0.5:
        print ("It's boleto |", np.round(i[0][0], 4), ' | ', str(f[n]))
        erros += 1
    else:
        print ("It's not a boleto |", np.round(i[0][0], 4), ' | ', str(f[n]))
    n += 1

precisao = ((n-erros)/(erros+(n-erros)))*100
print ('Mistakes:', erros)
print ('Got right:', n-erros)
print ('Accuracy:', precisao)

##############################################
##############################################   
resultados = []
precisões_epochs = []
for i in range(1, 18):
    classifier.load_weights(data_path + "\\classifier-{:02d}.hdf5".format(i))
    for n in range (1, len(f) + 1):
        imagem_teste = image.load_img('C:\\Users\\renat\\OneDrive\\Área de Trabalho\\classificaBoleto\\imagens\\{}'.format(f[n-1]),
                                      target_size = (192, 192),
                                      color_mode = 'grayscale')
        imagem_teste = image.img_to_array(imagem_teste)
        imagem_teste /= 255
        imagem_teste = np.expand_dims(imagem_teste, axis = 0)
        previsão = classifier.predict(imagem_teste)
        resultados.append(previsão)
        for y in range (1, len(f) + 1):
            resultados2.append(resultados)
        erros = 0 
        n = 0
        for i in resultados:
            if float(i[0][0]) > 0.5:
                erros += 1
            else:
                n += 1
        precisao = ((n-erros)/(erros+(n-erros)))*100
        
        