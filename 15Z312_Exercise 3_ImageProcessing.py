from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
import glob
import csv
import cv2
from numpy import array, asarray, ndarray

#training controls
batch_size = 25
epochs = 1
training_size = 0.7

# input image dimensions
img_rows, img_cols = 268, 182

# the data holders
x_test = []
x_train = []
y_test= []
y_train= []
tempY = []

#opening the dataset
dataset = csv.reader(open("csv/MovieGenre.csv",encoding="utf8",errors='replace'), delimiter=",")

#skipping the header line
next(dataset)

#the list of image files in SampleMoviePosters folder
flist=glob.glob('Movies/*.jpg')  

#extracting the data from the CSV file
for imdbId, Link, Title, Score, Genre, Poster in dataset:
    if(Score!=""):
        if(len((int(imdbId),float(Score)))==2):
            tempY.append((int(imdbId),float(Score)))


#setting the length of training data
length=int(len(flist)*training_size)

#extracting the data about the images that are available
i=0
for filename in flist:
    name=int(filename.split('\\')[-1][:-4])
    for z in tempY:
        if(z[0]==name):
            img = cv2.imread(filename)
            if(i<length):
                x_train.append(array(img))
                y_train.append(z[1])
            else:
                x_test.append(array(img))
                y_test.append(z[1])
    i+=1
    
#converting the data from lists to numpy arrays
x_train=asarray(x_train,dtype=float)
x_test=asarray(x_test,dtype=float)
y_train=asarray(y_train,dtype=float)
y_test=asarray(y_test,dtype=float)

#scaling down the RGB data
x_train /= 255
x_test /= 255

#printing stats about the features
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



#defining the model
model = Sequential()
#input
model.add(Conv2D(128,data_format = 'channels_last', kernel_size=(3, 3),
                 input_shape=(img_rows, img_cols,3)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
#convolutions

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

#dense layers
model.add(Flatten())

model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.15))

model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.15))

model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.15))

#output
model.add(Dense(1))

#printing model summary
print(model.summary())

#compiling the model
model.compile(optimizer='RMSprop', loss='mse', metrics=['mae'])

#training the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#testing the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])