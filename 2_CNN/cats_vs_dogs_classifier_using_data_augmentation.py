from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout

batch_size = 16
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# only one transformation on test bcz we cant apply augmentaion on test data only on train
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './train',
    target_size=(150,150),
    batch_size=batch_size,
    class_mode='binary',) # since we use binary crossentropy we need binary labels

validation_generator = test_datagen.flow_from_directory(
    './val',
    target_size=(150,150),
    batch_size=batch_size,
    class_mode='binary',)

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3), activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation='relu',))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# remember this while generating data
# we dont use fit we use fit_generator
# instead of x we give generator para
model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=800 // batch_size)



                 

