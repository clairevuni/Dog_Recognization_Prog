import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from glob import glob
from model import create_dog_breed_classifier_model, get_dog_names

def load_and_preprocess_image(image_path, target_size=(299, 299)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to between 0 and 1
    return img_array

def predict_dog_breed(image_path, model, dog_names, threshold=0.4):
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_dog_breed = dog_names[predicted_class_index]
    predicted_probability = predictions[0][predicted_class_index]
    
    if predicted_probability >= threshold:
        return predicted_dog_breed, predictions
    else:
        return "Non è un cane", predictions

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image_path.set(file_path)
        show_selected_image(file_path)

def show_selected_image(file_path):
    img = Image.open(file_path)
    img = img.resize((250, 250), Image.BICUBIC)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img


def predict():
    if image_path.get():
        predicted_breed, predictions = predict_dog_breed(image_path.get(), model, dog_names)
        
        threshold = 0.4  # Impostiamo il threshold a 0.4 o qualsiasi altro valore desiderato
        
        is_dog = predicted_breed != "Non è un cane" and predictions[0][dog_names.index(predicted_breed)] >= threshold
        
        if is_dog:
            predicted_breed_label.config(text=f"Predicted Dog Breed: {predicted_breed}", font=("Arial", 14, "bold"), foreground="#2E8B57")
            predictions_text.config(state=tk.NORMAL)
            predictions_text.delete("1.0", tk.END)

            top_indices = np.argsort(predictions.ravel())[::-1][:3]

            for idx in top_indices:
                breed = dog_names[idx]
                prob = predictions.ravel()[idx] * 100
                predictions_text.insert(tk.END, f"{breed}: {prob:.2f}%\n", "prediction")

            predictions_text.tag_config("prediction", foreground="#2E8B57")
            predictions_text.config(state=tk.DISABLED)
        else:
            predicted_breed_label.config(text="Non è un cane", font=("Arial", 14, "bold"), foreground="#FF0000")
            predictions_text.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Dog Breed Prediction")
root.configure(background="#f0f0f0") 

model = load_model("dog_breed_classifier_model_with_dropout_trained.h5")

base_folder = "C:/Users/39334/Desktop/MULTIMEDIA MAG/Dog_Recognization"
train_folder = os.path.join(base_folder, "train")
dog_names = get_dog_names(train_folder)

image_path = tk.StringVar()

select_button = tk.Button(root, text="Select Image", command=select_image, bg="#2E8B57", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT)
select_button.pack(pady=10)

selected_image_label = tk.Label(root, text="Selected Image:", bg="#f0f0f0", font=("Arial", 12))
selected_image_label.pack()

panel = tk.Label(root)
panel.pack()

predict_button = tk.Button(root, text="Predict", command=predict, bg="#2E8B57", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT)
predict_button.pack(pady=10)

predicted_breed_label = tk.Label(root, text="", bg="#f0f0f0", font=("Arial", 14, "bold"))
predicted_breed_label.pack()

predictions_text = tk.Text(root, height=10, width=50, bg="#f0f0f0", font=("Arial", 12))
predictions_text.pack(pady=10)
predictions_text.config(state=tk.DISABLED)

root.mainloop()
