''' image tools as others '''
import os 
import matplotlib.image as img
import numpy as np
from tensorflow.keras.preprocessing import image

def raw_to_tf_format(filename,size_tuple):
    if (os.path.isfile(filename)):
        img = image.load_img(filename,target_zise=size_tuple)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array,axis=0)
        
        return np.vstack([img_array])