from transformers import ViTForImageClassification, TrainingArguments, Trainer, ViTImageProcessor
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset = load_dataset("split_ttv_dataset_type_of_plants/Train_Set_Folder")
dataset_valid = load_dataset("split_ttv_dataset_type_of_plants/Validation_Set_Folder")
print(dataset)

model_name = "google/vit-base-patch16-224"
# load and inspecting data
train_data = dataset["train"]
class_names = dataset["train"].features["label"].names
print("Class names: ", class_names)
print(dataset["train"].features)

for i in range(3):
    plt.imshow(dataset["train"][i]["image"])
    plt.title(class_names[dataset["train"][i]["label"]])
    plt.show()



feature_extractor = ViTImageProcessor.from_pretrained(model_name)
print("transform function")
# processing the dataset
def transform(examples):
    try:
        examples["pixel_values"] = feature_extractor(
        images=examples["image"],
        return_tensors="pt"
        )["pixel_values"]
    except Exception as e:
        print(f"Error processing image: {e}")
        examples["pixel_values"] = None
    return examples




print("setting up models")
dataset = dataset.map(transform, batched=True, num_proc=1)
dataset.set_format(type="torch", columns=["pixel_values", "label"])
dataset_valid = dataset_valid.map(transform, batched=True, num_proc=1)
dataset_valid.set_format(type="torch", columns=["pixel_values", "label"])

print("right here")
print("Sample pixel_values shape:", dataset["train"][0]["pixel_values"].shape)
print("Sample label:", dataset["train"][0]["label"])
try:
    print("YEAH")
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=len(class_names),
        ignore_mismatched_sizes=True
    )
    print("MODEL LOADED WITH LABELS: ", model.config.num_labels)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

print("after here")
print(dataset.keys())
print(dataset_valid.keys())


training_args = TrainingArguments(
    output_dir="./vit-plant-identification",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset_valid["train"],
)
print("training")
trainer.train()

trainer.save_model("./vit-plant-identification")
feature_extractor.save_pretrained("./vit-plant-identification")
print("Model and feature extractor saved to ./vit-plant-identification")