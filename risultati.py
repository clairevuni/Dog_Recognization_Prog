from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_image(image_path, target_size=(299, 299)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizzazione tra 0 e 1
    return img_array

def load_validation_data(validation_folder, target_size=(299, 299)):
    file_paths = glob(os.path.join(validation_folder, "*", "*.jpg"))
    images = []
    labels = []
    for file_path in file_paths:
        label = os.path.basename(os.path.dirname(file_path))
        image_array = load_and_preprocess_image(file_path, target_size)
        images.append(image_array)
        labels.append(label)
    return np.vstack(images), np.array(labels)

if __name__ == "__main__":
    # Caricamento delle immagini di validazione
    valid_folder = "C:/Users/39334/Desktop/MULTIMEDIA MAG/Dog_Recognization/valid"
    X_validation, y_validation = load_validation_data(valid_folder)

    # Path ai modelli addestrati
    model_path1 = "dog_breed_classifier_model.h5"
    model_path2 = "dog_breed_classifier_model.h5"

    # Caricamento dei modelli addestrati
    model1 = load_model(model_path1)  
    model2 = load_model(model_path2)

    # Valutazione dei modelli sui dati di validazione
    evaluation1 = model1.evaluate(X_validation, y_validation)
    evaluation2 = model2.evaluate(X_validation, y_validation)

    # Stampa dei risultati
    print("Risultati del Modello 1:")
    print("Loss:", evaluation1[0])
    print("Accuracy:", evaluation1[1])

    print("\nRisultati del Modello 2 (con rimozione del rumore):")
    print("Loss:", evaluation2[0])
    print("Accuracy:", evaluation2[1])

    # Confronto delle accuratezze per determinare quale modello è migliore
    if evaluation1[1] > evaluation2[1]:
        print("\nIl Modello 1 è migliore in termini di accuratezza.")
    elif evaluation1[1] < evaluation2[1]:
        print("\nIl Modello 2 (con rimozione del rumore) è migliore in termini di accuratezza.")
    else:
        print("\nI due modelli hanno la stessa accuratezza.")

    # Grafici
    history1 = load_model(model_path1).history.history
    history2 = load_model(model_path2).history.history

    # Loss plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history1['loss'], label='Modello 1 Loss (train)')
    plt.plot(history1['val_loss'], label='Modello 1 Loss (val)')
    plt.plot(history2['loss'], label='Modello 2 Loss (train)')
    plt.plot(history2['val_loss'], label='Modello 2 Loss (val)')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuratezza plot
    plt.subplot(1, 2, 2)
    plt.plot(history1['accuracy'], label='Modello 1 Accuracy (train)')
    plt.plot(history1['val_accuracy'], label='Modello 1 Accuracy (val)')
    plt.plot(history2['accuracy'], label='Modello 2 Accuracy (train)')
    plt.plot(history2['val_accuracy'], label='Modello 2 Accuracy (val)')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
