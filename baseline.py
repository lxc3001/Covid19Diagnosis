
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from contextlib import redirect_stdout


from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertModel, AutoModel
from torchvision.models import resnet50
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from collections import Counter

from convirt import ConVIRTImageEncoder, ConVIRTTextEncoder
from sequentialConvirt import TemporalModel
from contrastiveHierLoss import MultimodalEncoder
from classificationDataset import ClassificationDataset, label_to_idx, process_record

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------- Baseline (Linear Probe) -----------------
num_classes = 7

def run_baseline(baseline):
    if baseline == "baseline_1":
        print("Running Baseline 1: Random Initialization")
        image_encoder = resnet50(pretrained=False)
        image_encoder.fc = nn.Identity()
        text_encoder = BertModel.from_pretrained('bert-base-uncased')
        text_encoder.init_weights()
    elif baseline == "baseline_2":
        print("Running Baseline 2: ImageNet Pretrained + ClinicalBERT")
        image_encoder = resnet50(pretrained=True)
        image_encoder.fc = nn.Identity()
        text_encoder = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    # Freeze encoders
    for param in image_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    print("Baseline setup complete")
    return image_encoder, text_encoder

# ----------------- Load Pretrained Convirt, SequentialConvirt, HierModel ----------------- 
def load_convirt(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_encoder = ConVIRTImageEncoder(output_dim=128)
    text_encoder = ConVIRTTextEncoder(output_dim=128)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    image_encoder.load_state_dict(checkpoint['image_encoder_state_dict'])
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

    return image_encoder, text_encoder


def load_sequential_encoders(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TemporalModel(img_output_dim=256, txt_output_dim=256, lstm_hidden_dim=256, pretrained_text_model="medicalai/ClinicalBERT").to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model.image_encoder, model.text_encoder


def load_hier_encoders(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalEncoder(image_output_dim=256, text_output_dim=256)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model.image_encoder, model.text_encoder


def load_pretrained_encoders(model_type, checkpoint_path):
    if model_type == "convirt":
        return load_convirt(checkpoint_path)
    elif model_type == "sequential_convirt":
        return load_sequential_encoders(checkpoint_path)
    elif model_type == "hier_model":
        return load_hier_encoders(checkpoint_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ----------------- Train + Test -----------------
def get_class_weights(labels_list, num_classes=7, device='cpu'):
    all_classes = np.arange(num_classes)

    present_classes = np.unique(labels_list)

    if set(present_classes) == set(all_classes):
        class_weights = compute_class_weight(class_weight='balanced', classes=all_classes, y=labels_list)
    else:
        partial_weights = compute_class_weight(class_weight='balanced', classes=present_classes, y=labels_list)
        class_weights = np.zeros(num_classes, dtype=np.float32)
        for idx, cls in enumerate(present_classes):
            class_weights[cls] = partial_weights[idx]
    
    return torch.tensor(class_weights, dtype=torch.float).to(device)



def train_and_evaluate(
    model_type,
    train_loader,
    test_loader,
    checkpoint_path=None,
    baseline=False,
    feature_dim=2048,
    num_epochs=10,
    lr=1e-3,
    use_text=False 
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Load Encoder --------
    if baseline:
        image_encoder, text_encoder = run_baseline(model_type)
    else:
        image_encoder, text_encoder = load_pretrained_encoders(model_type, checkpoint_path)


    image_encoder = image_encoder.to(device)
    if text_encoder:
        text_encoder = text_encoder.to(device)

    for param in image_encoder.parameters():
        param.requires_grad = False
    if text_encoder:
        for param in text_encoder.parameters():
            param.requires_grad = False

    # -------- Adjust feature_dim --------
    if use_text:
        if model_type == "convirt":
            feature_dim = 128 + 128
        elif baseline:
            feature_dim = 2048 + 768
        else:
            feature_dim = 256 + 256
    else:
        if model_type == "convirt":
            feature_dim = 128
        elif baseline:
            feature_dim = 2048
        else:
            feature_dim = 256

    # -------- Classifier --------
    classifier = nn.Linear(feature_dim, 7)
    nn.init.xavier_uniform_(classifier.weight)
    nn.init.zeros_(classifier.bias)
    classifier = classifier.to(device)

    labels_all = train_loader.dataset.dataframe['final_label'].tolist()
    class_weights = get_class_weights(labels_all, num_classes=7, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    # -------- Train --------
    for epoch in range(num_epochs):
        image_encoder.eval()
        if text_encoder:
            text_encoder.eval()
        classifier.train()

        total_loss, total_correct, total_samples = 0, 0, 0

        for batch in train_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                img_features = image_encoder(images)
                if use_text and text_encoder is not None:
                    text_features = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                    if hasattr(text_features, 'last_hidden_state'):
                        text_features = text_features.last_hidden_state[:, 0, :]
                    features = torch.cat([img_features, text_features], dim=1)
                else:
                    features = img_features

            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")

    # -------- Test --------
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            img_features = image_encoder(images)
            if use_text and text_encoder is not None:
                text_features = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(text_features, 'last_hidden_state'):
                    text_features = text_features.last_hidden_state[:, 0, :]
                features = torch.cat([img_features, text_features], dim=1)
            else:
                features = img_features

            logits = classifier(features)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    label_counter = Counter(all_labels)
    pred_counter = Counter(all_preds)
    print(f"Ground Truth Distribution: {label_counter}")
    print(f"Predicted Distribution:    {pred_counter}")

    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"[{model_type}] Test F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

    return f1, accuracy




class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() 

    def flush(self):
        for f in self.files:
            f.flush()



if __name__ == "__main__":
    log_file = open("experiment_img_log.txt", "w", encoding="utf-8")
    tee = Tee(sys.__stdout__, log_file)
    sys.stdout = tee

    csv_path = './data/test_data.csv'
    img_root = './covid-chestxray-dataset-master/data/remaining_images'
    dataframe = pd.read_csv(csv_path).reset_index(drop=True)

    dataframe['final_label'] = dataframe['finding'].apply(process_record)
    dataframe['final_label'] = dataframe['final_label'].map(label_to_idx)
    dataframe = dataframe.dropna(subset=['final_label'])
    dataframe['final_label'] = dataframe['final_label'].astype(int)

    N_ROUNDS = 5  # Experiment rounds
    all_results = []

    models_info = [
        {"model_type": "baseline_1", "baseline": True, "tokenizer": "bert-base-uncased"},
        {"model_type": "baseline_2", "baseline": True, "tokenizer": "emilyalsentzer/Bio_ClinicalBERT"},
        {"model_type": "convirt", "path": "./convirt_checkpoints/convirt_epoch_20.pth", "feature_dim": 256, "tokenizer": "bert-base-uncased"},
        {"model_type": "sequential_convirt", "path": "./temporal_checkpoints/temporal_epoch_20.pth", "feature_dim": 512, "tokenizer": "medicalai/ClinicalBERT"},
        {"model_type": "hier_model", "path": "./hier_checkpoints/epoch_20.pth", "feature_dim": 512, "tokenizer": "emilyalsentzer/Bio_ClinicalBERT"},
    ]

    for round_idx in range(N_ROUNDS):
        print(f"\n============== Round {round_idx+1}/{N_ROUNDS} ==============")
        train_df, test_df = train_test_split(dataframe, test_size=0.2, stratify=dataframe['final_label'], random_state=42 + round_idx)

        for model_info in models_info:
            model_type = model_info["model_type"]
            print(f"\n========== Running {model_type} ==========")

            tokenizer_name = model_info["tokenizer"]
            train_dataset = ClassificationDataset(train_df, img_root, tokenizer_name=tokenizer_name)
            test_dataset = ClassificationDataset(test_df, img_root, tokenizer_name=tokenizer_name)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

            f1, acc = train_and_evaluate(
                model_type=model_type,
                checkpoint_path=model_info.get("path"),
                train_loader=train_loader,
                test_loader=test_loader,
                num_epochs=10,
                lr=1e-3,
                use_text=False, # If using text features
                baseline=model_info.get("baseline", False) # If using baseline models
            )

            all_results.append({
                "Round": round_idx + 1,
                "Model": model_type,
                "F1 Score": f1,
                "Accuracy": acc
            })

    # Save all rounds results
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv("classification_img_results.csv", index=False)
    print("\nSaved all experiment results to classification_img_results.csv")

    sys.stdout = sys.__stdout__
    log_file.close()

