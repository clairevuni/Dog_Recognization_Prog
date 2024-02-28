from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import sys
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6]) 
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))  

def get_boxes(obj):
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    inputs = processor(text=obj, images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])
    
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {obj} with confidence {round(score.item(), 3)} at location {box}")

    boxes = boxes.tolist()
    if len(boxes) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        for box in boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')
        plt.title("Detected objects", fontsize=20)
        plt.show()

    
    input_boxes = torch.tensor(boxes, device=model.device)
    labels = [1]*len(input_boxes)
    return input_boxes, labels


image_path = input("Enter the path to the image containing a dog: ")

if not os.path.exists(image_path):
    print("Error: Invalid image path.")
    sys.exit(1)

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

sam_checkpoint = "checkpoints\sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)


def save_image(image, filename, save_directory):
    os.makedirs(save_directory, exist_ok=True)  
    filepath = os.path.join(save_directory, filename)
    image.save(filepath) 


obj = "dog"
input_boxes, input_label = get_boxes(obj)

if len(input_boxes) == 0:
    print("No dog found") #nessun cane
    
else:
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

    for i, box in enumerate(input_boxes):
        input_box = np.array(box.tolist())
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        image_copy = image.copy()
        x0, y0, x1, y1 = map(int, input_box)
        cropped_image = Image.fromarray(image_copy[y0:y1, x0:x1])
        plt.figure(figsize=(10, 10))
        plt.imshow(cropped_image)
        plt.axis('off')
        plt.show()
        
    save_option = input("Do you want to save the image? (y/n): ")
    if save_option.lower() == 'y':
        save_directory = input("Enter the directory to save the image: ")
        filename = f"image_{i+1}.png"
        save_image(cropped_image, filename, save_directory)