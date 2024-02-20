from tensorflow.keras import applications
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt

def create_dog_breed_classifier_model(img_width, img_height, channels, num_classes):
    # Load InceptionV3 architecture with fully connected layers for dog breed classification
    base_model = applications.InceptionV3(input_shape=(299, 299, 3), weights='imagenet', include_top=False)

    # Freeze all pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add a fully connected layer for your classification
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))  # Add Dropout layer with dropout rate of 0.5
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    img_width, img_height = 299, 299  # Update input shape
    channels = 3

    # List of dog breed names
    base_folder = "C:/Users/39334/Desktop/MULTIMEDIA MAG/Dog_Recognization"
    train_folder = os.path.join(base_folder, "train")
    dog_names = [item.split(os.sep)[-1] for item in sorted(glob(os.path.join(train_folder, "*")))]

    num_classes = len(dog_names)  # Number of classes is the number of dog breeds

    # Create the model with dropout
    model = create_dog_breed_classifier_model(img_width, img_height, channels, num_classes)

    # Display the model architecture
    model.summary()

    # Create data generators for training and validation
    batch_size = 32
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

    # Setup History callback to record metrics during training
    history = model.fit(train_generator, epochs=2, validation_data=validation_generator)


    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.ylim([0, 0.99])  # Set the y-axis limit to 0.99
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    print("Model trained and saved successfully.")
