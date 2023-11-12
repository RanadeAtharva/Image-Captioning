#!/usr/bin/env python
# coding: utf-8

# In[34]:



from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.models import load_model, Model
from tensorflow.keras.preprocessing.text import Tokenizer

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd


import warnings
warnings.filterwarnings("ignore")



model = load_model("model_19.h5") ## Threshold = 10

model_temp = VGG16(weights="imagenet", input_shape=(224,224,3))

# Create a new model, by removing the last layer (output layer of 1000 classes) from the resnet50
model_vgg = Model(model_temp.input, model_temp.layers[-2].output)




# Load the word_to_idx and idx_to_word from disk
word_to_index = {}
with open ("word_to_idx.pkl", 'rb') as file:
    word_to_index = pd.read_pickle(file, compression=None)

index_to_word = {}
with open ("idx_to_word.pkl", 'rb') as file:
    index_to_word = pd.read_pickle(file, compression=None)




def predict_caption(photo):

     inp_text = "startseq"

     #max_len = 80 which is maximum length of caption
     for i in range(80):
         sequence = [word_to_index[w] for w in inp_text.split() if w in word_to_index]
         sequence = pad_sequences([sequence], maxlen=80, padding='post')

         ypred = model.predict([photo, sequence])
         ypred = ypred.argmax()
         word = index_to_word[ypred]

         inp_text += (' ' + word)

         if word == 'endseq':
             break

     final_caption = inp_text.split()[1:-1]
     final_caption = ' '.join(final_caption)
     return final_caption



def preprocess_image (img):
     img = image.load_img(img, target_size=(224, 224))
     img = image.img_to_array(img)

     # Convert 3D tensor to a 4D tendor
     img = np.expand_dims(img, axis=0)

     #Normalize image accoring to ResNet50 requirement
     img = preprocess_input(img)

     return img


 # A wrapper function, which inputs an image and returns its encoding (feature vector)
def encode_image (img):
     img = preprocess_image(img)

     feature_vector = model_vgg.predict(img)
     # feature_vector = feature_vector.reshape((-1,))
     return feature_vector


def runModel(img_name):
     #img_name = input("enter the image name to generate:\t")

     print("Encoding the image ...")
     photo = encode_image(img_name).reshape((1, 4096))



     print("Running model to generate the caption...")
     caption = predict_caption(photo)

     img_data = plt.imread(img_name)
     plt.imshow(img_data)
     plt.axis("off")

     #plt.show()
     print(caption)
     return caption
