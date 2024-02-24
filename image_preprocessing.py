from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import random

def load_and_preprocess_image(image_path, target_size=(299, 299), normalize=True, random_crop=False, noise=False, noise_factor=0.1):
    img = load_img(image_path, target_size=target_size)
    

    img_array = img_to_array(img)
    
    if random_crop:
        x_offset = random.randint(0, img_array.shape[1] - target_size[1])
        y_offset = random.randint(0, img_array.shape[0] - target_size[0])
        img_array = img_array[y_offset:y_offset+target_size[0], x_offset:x_offset+target_size[1], :]
    
    if noise:
        img_array += np.random.normal(loc=0.0, scale=noise_factor, size=img_array.shape)
        img_array = np.clip(img_array, 0.0, 255.0)

    img_array = np.expand_dims(img_array, axis=0)
    
    if normalize:
        img_array /= 255.0
    
    return img_array
