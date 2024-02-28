#primofile.py
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from numpy import expand_dims
import numpy as np
from glob import glob
from sklearn.datasets import load_files
import os

img_width, img_height = 224, 224
batch_size = 32

base_folder = "C:/Users/39334/Desktop/MULTIMEDIA MAG/Dog_Recognization"
train_folder, valid_folder, test_folder = [os.path.join(base_folder, folder) for folder in ["train", "valid", "test"]]

# 
def load_dataset(path):
    data = load_files(path)
    return np.array(data['filenames']), to_categorical(np.array(data['target']), len(os.listdir(path)))

train_files, train_targets = load_dataset(train_folder)
valid_files, valid_targets = load_dataset(valid_folder)
test_files, test_targets = load_dataset(test_folder)


dog_names = [item.split(os.sep)[-1] for item in sorted(glob(os.path.join(train_folder, "*")))]


print('Total dog categories:', len(dog_names))
print('Total dog images across all sets:', len(np.hstack([train_files, valid_files, test_files])))
print('Training dog images:', len(train_files))
print('Validation dog images:', len(valid_files))
print('Test dog images:', len(test_files))


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2
)

valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# i generatori di dati
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=1337
)

valid_generator = valid_datagen.flow_from_directory(
    train_folder,
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=1337
)

num_classes = len(train_generator.class_indices)
train_labels = train_generator.classes
train_labels = to_categorical(train_labels, num_classes=num_classes)
valid_labels = valid_generator.classes
valid_labels = to_categorical(valid_labels, num_classes=num_classes)
nb_train_samples = len(train_generator.filenames)
nb_valid_samples = len(valid_generator.filenames)


img = load_img('C:\\Users\\39334\\Desktop\\MULTIMEDIA MAG\\Dog_Recognization\\benny.jpg')
data = img_to_array(img)
samples = expand_dims(data, 0)
it = train_datagen.flow(samples, batch_size=1)


plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    batch = it.next()
    image = batch[0]
    plt.imshow(image)
    plt.axis('off')

plt.savefig('all_augmented_images.png', transparent=False, bbox_inches='tight', dpi=900)
plt.show()

import csv

train_labels_list = train_labels.tolist()


file_label_tuples = [(file, label) for file, label in zip(train_files, train_labels_list)]

#  CSV
csv_file_path = 'labels.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['File', 'Label'])
    writer.writerows(file_label_tuples)

print(f"File CSV creato con successo: {csv_file_path}")
