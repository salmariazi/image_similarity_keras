from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as  np
from keras.applications.vgg16 import VGG16
from keras.applications import vgg16
from keras import models, Model
from annoy import AnnoyIndex
import pandas as pd
from scipy import spatial

import ssl
import os 


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder,filename))
        img = load_img(os.path.join(folder,filename),  target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape((1,) + img.shape)
        if img is not None:
            images.append(img)
    return images

def get_all_images():
    images1 = load_images_from_folder('geological_similarity/andesite/')
    images2 = load_images_from_folder('geological_similarity/gneiss/')
    images3 = load_images_from_folder('geological_similarity/marble/')
    images4 = load_images_from_folder('geological_similarity/quartzite/')
    images5 = load_images_from_folder('geological_similarity/rhyolite/')
    images6 = load_images_from_folder('geological_similarity/schist/')
    all_imgs_arr = np.array([images1+images2+images3+images4+images5+images6])
    return all_imgs_arr




def create_model():
    # loading vgg16 model and using all the layers until the 2 to the last to use all the learned cnn layers

    ssl._create_default_https_context = ssl._create_unverified_context
    vgg = VGG16(include_top=True)
    model2 = Model(vgg.input, vgg.layers[-2].output)
    model2.save('vgg_4096.h5') # saving the model just in case
    return model2

def get_preds(all_imgs_arr):
    preds_all = np.zeros((len(all_imgs_arr),4096))
    for j in range(all_imgs_arr.shape[0]):
        preds_all[j] = model.predict(all_imgs_arr[j])
        
    return preds_all


if __name__ == '__main__':
    all_imgs_arr = get_all_images()
    all_imgs_arr = all_imgs_arr.reshape(all_imgs_arr.shape[1], 1, 224, 224, 3)
    np.save('all_images', all_imgs_arr)
    model = create_model()
    preds_all = get_preds(all_imgs_arr)
    np.savez('images_preds', images=all_imgs_arr, preds=preds_all)
