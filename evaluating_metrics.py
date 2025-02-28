from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate

model_path = "./vit-plant-identification"

model = ViTForImageClassification.from_pretrained(model_path)
feature_extractor = ViTImageProcessor.from_pretrained(model_path)

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

dataset = load_dataset("split_ttv_dataset_type_of_plants/Validation_Set_Folder")
dataset = dataset.map(transform, batched=True, num_proc=1)
dataset.set_format(type="torch", columns=["pixel_values", "label"])

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)  # Get the predicted class (index with the highest logit)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./vit-plant-identification",
    per_device_eval_batch_size=8,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset["train"],  # Use validation set
    compute_metrics=compute_metrics
)

eval_results = trainer.evaluate()
print(eval_results)
