from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

img = image.load_img('./train/cat.jpg', target_size=(200,200))

datagen = ImageDataGenerator(
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='constant'
    )

img = image.img_to_array(img)
print(img.shape)

input_batch = img.reshape(1,200,200,3)

i=0
# flow --- for single img
# flow_from_dir ---- multiple img in dir
for output in datagen.flow(input_batch, batch_size=1, save_to_dir='aug'):
    i+=1
    if i==10:
        break



#plt.imshow(img)
#print(type(img))
#plt.show()

