
# coding: utf-8

# In[ ]:


from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img,img_to_array
from sklearn.metrics import mean_squared_error
from keras.initializers import RandomNormal
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD, Adam
from keras.models import Model,Sequential
from keras.layers import *
from keras import backend as K
from keras.models import model_from_json
from matplotlib import cm as CM
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import scipy.io as io
from PIL import Image
import PIL
import h5py
import os
import glob
import cv2
import random
import math
import sys


# In[ ]:


K.clear_session()
root = 'data'


# In[ ]:


part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
temp = 'test_images'
path_sets = [part_B_train]


# In[ ]:


img_paths = []

for path in path_sets:
    
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        
        img_paths.append(str(img_path))
        
print("Total images : ",len(img_paths))


# In[ ]:


from keras.applications.vgg16 import preprocess_input as k_preprocess_input
from keras.preprocessing import image as k_image

def create_img(path):
    #Function to load,normalize and return image 
    im = k_image.load_img(path, target_size=(768, 1024))
    im = k_image.img_to_array(im)
    im = k_preprocess_input(im)
    return im

def get_input(path):
    path = path[0] 
    img = create_img(path)
    return(img)
    
def get_output(path):
    gt_file = h5py.File(path,'r')
    target = np.asarray(gt_file['density'])
    target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
    target = np.expand_dims(target,axis  = 3)
    
    return target


# In[ ]:


#Image data generator 
def image_generator(files, batch_size = 16):
    
    while True:
        
        input_path = np.random.choice(a = files, size = batch_size)
        
        batch_input = []
        batch_output = [] 
          
        #for input_path in batch_paths:
        inputt = get_input(input_path)
        output = get_output(input_path[0].replace('.jpg','.h5').replace('images','ground_truth') )
            
       
        batch_input += [inputt]
        batch_output += [output]
    

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        
        yield(batch_x, batch_y)


# In[ ]:


def save_mod(model , str1 , str2):
    model.save_weights(str1)
    model_json = model.to_json()
    with open(str2, "w") as json_file:
        json_file.write(model_json)


# In[ ]:


def init_weights_vgg(model):
    vgg =  VGG16(weights='imagenet', include_top=False)
    
    #json_file = open('models/VGG_16.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    #loaded_model.load_weights("weights/VGG_16.h5")
    
    #vgg = loaded_model
    
    vgg_weights=[]
    for layer in vgg.layers:
        if('conv' in layer.name):
            vgg_weights.append(layer.get_weights())
    
    
    offset=0
    i=0
    while(i<10):
        if('conv' in model.layers[i+offset].name):
            model.layers[i+offset].set_weights(vgg_weights[i])
            i=i+1
            #print('h')
            
        else:
            offset=offset+1

    return (model)
    


# In[ ]:


def maaae(y_true, y_pred):
    return abs(K.sum(y_true) - K.sum(y_pred))
def mssse(y_true, y_pred):
    return (K.sum(y_true) - K.sum(y_pred)) * (K.sum(y_true) - K.sum(y_pred))


# In[ ]:


def euclidean_distance_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)


# In[ ]:


# Neural network model : VGG + Conv
def CrowdNet():  
    #Variable Input Size
    rows = None
    cols = None

    #Batch Normalisation option

    batch_norm = None
    kernel = (3, 3)
    init = RandomNormal(stddev=0.01)
    model = Sequential() 

    #custom VGG:
    if(batch_norm):
        model.add(Conv2D(64, kernel_size = kernel, input_shape = (rows,cols,3),activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=2))            
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same'))
        model.add(BatchNormalization())

    else:
        model.add(Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same',input_shape = (rows, cols, 3), kernel_initializer = init))
        model.add(Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(MaxPooling2D(strides=2))
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same', kernel_initializer = init))
        model.add(MaxPooling2D(strides=2))            
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))
        model.add(Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same', kernel_initializer = init))




    #Conv2D
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(256, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(128, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation='relu', dilation_rate = 2, kernel_initializer = init, padding = 'same'))
    model.add(Conv2D(1, (1, 1), activation='relu', dilation_rate = 1, kernel_initializer = init, padding = 'same'))

    model = init_weights_vgg(model)
    # sgd = SGD(lr = 1e-7, decay = (5*1e-4), momentum = 0.95)
    model.compile(optimizer=Adam(1e-5), loss='mse', metrics=[maaae, mssse])

    return model


# In[ ]:


model = CrowdNet()


# In[ ]:


train_gen = image_generator(img_paths, 1)


# In[ ]:


part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_B_test]


# In[ ]:


def create_test_img(path):
    im = k_image.load_img(path, target_size=(768, 1024))
    im = k_image.img_to_array(im)
    im = k_preprocess_input(im)
    im = np.expand_dims(im, axis=0)
    return im


# In[ ]:


from sklearn.metrics import mean_absolute_error
best_mae = 1000
be = 0 
for i in range(1, 400):
    print('Fitting', str(i), 'epoch')
    model.fit_generator(train_gen,epochs=1, steps_per_epoch=700, verbose=1)
    
    if i%3 == 0:
        print('Evaluation with test set!')
        name = []
        y_true = []
        y_pred = []
        for image in img_paths:
            name.append(image)
            gt = h5py.File(image.replace('.jpg','.h5').replace('images','ground_truth'))
            groundtruth = np.asarray(gt['density'])
            num1 = np.sum(groundtruth)
            y_true.append(np.sum(num1))
            img = create_test_img(image)
            num = np.sum(model.predict(img))
            y_pred.append(np.sum(num))
        now_mae = mean_absolute_error(np.array(y_true),np.array(y_pred))
        if now_mae < best_mae:
            save_mod(model, "weights/model_B_400_adam_weights.h5","models/Model_adam.json")
            be = i; best_mae = now_mae
        print('Best MAE:', best_mae, 'Best Ep:', be)

