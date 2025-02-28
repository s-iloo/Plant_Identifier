from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from datasets import load_dataset
import torch

model = ViTForImageClassification.from_pretrained("./vit-plant-identification")
feature_extractor = ViTImageProcessor.from_pretrained("./vit-plant-identification")

# class_names = list(model.config.id2label.values())
dataset = load_dataset("split_ttv_dataset_type_of_plants/Train_Set_Folder")
class_names = dataset["train"].features["label"].names
def predict_plant_type(image_path):
    image = Image.open(image_path)
    #preprocess image
    inputs = feature_extractor(images=image, return_tensors="pt")

    # performing the inference
    with torch.no_grad(): 
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=-1).item()
    predicted_class_name = class_names[predicted_class]

    return predicted_class_name

if __name__ == "__main__":
    image_path = input("Enter the path to the plant image or a URL: ")

    try: 
        predicted_plant = predict_plant_type(image_path)
        print(f"The predicted plant type is: {predicted_plant}")
    except Exception as e:
        print(f"Error: {e}")