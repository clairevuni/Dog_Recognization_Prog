from tensorflow.keras import applications
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt
from glob import glob

def create_dog_breed_classifier_model(img_width, img_height, channels, num_classes):
    base_model = applications.InceptionV3(input_shape=(299, 299, 3), weights='imagenet', include_top=False)

    #
    for layer in base_model.layers:
        layer.trainable = False

    # 
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))  # Dropout
    model.add(Dense(num_classes, activation='softmax'))

    # 
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def get_dog_names(train_folder):
    dog_names = [item.split(os.sep)[-1] for item in sorted(glob(os.path.join(train_folder, "*")))]
    return dog_names


def train_dog_breed_classifier_model(model, train_generator, validation_generator, epochs):
    
    checkpoint = ModelCheckpoint("dog_breed_classifier_model_with_dropout_best.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    # addestramento
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[checkpoint])
    
    # P
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return history

if __name__ == "__main__":
    
    img_width, img_height = 299, 299 
    channels = 3
    batch_size = 32
    epochs = 10
    
    base_folder = "C:/Users/39334/Desktop/MULTIMEDIA MAG/Dog_Recognization"
    train_folder = os.path.join(base_folder, "train")
    dog_names = [item.split(os.sep)[-1] for item in sorted(glob(os.path.join(train_folder, "*")))]

    num_classes = len(dog_names) 

    
    model = create_dog_breed_classifier_model(img_width, img_height, channels, num_classes)
    model.summary()
    
   
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

    validation_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=1337
    )

    train_dog_breed_classifier_model(model, train_generator, validation_generator, epochs)