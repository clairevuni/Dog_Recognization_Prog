import os
import shutil
import random

# Percorsi delle cartelle
base_folder = "C:/Users/39334/Desktop/MULTIMEDIA MAG/Dog_Recognization"
images_folder = os.path.join(base_folder, "images", "Images")
train_folder = os.path.join(base_folder, "train")
valid_folder = os.path.join(base_folder, "valid")
test_folder = os.path.join(base_folder, "test")

# Lista delle classi
classes = os.listdir(images_folder)

# Creazione delle cartelle train, valid, e test
for folder in [train_folder, valid_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)

# Suddivisione delle immagini in train, valid, e test
for dog_class in classes:
    class_folder = os.path.join(images_folder, dog_class)
    class_images = os.listdir(class_folder)
    random.shuffle(class_images)

    train_split = int(0.7 * len(class_images))
    valid_split = int(0.15 * len(class_images))

    train_images = class_images[:train_split]
    valid_images = class_images[train_split:train_split + valid_split]
    test_images = class_images[train_split + valid_split:]

    # Creazione delle sottocartelle nelle cartelle train, valid, e test
    for folder in [train_folder, valid_folder, test_folder]:
        dog_class_folder = os.path.join(folder, dog_class)
        os.makedirs(dog_class_folder, exist_ok=True)

    # Copia delle immagini nelle sottocartelle train, valid, e test
    for image in train_images:
        shutil.copy(
            os.path.join(class_folder, image),
            os.path.join(train_folder, dog_class, image)
        )

    for image in valid_images:
        shutil.copy(
            os.path.join(class_folder, image),
            os.path.join(valid_folder, dog_class, image)
        )

    for image in test_images:
        shutil.copy(
            os.path.join(class_folder, image),
            os.path.join(test_folder, dog_class, image)
        )
