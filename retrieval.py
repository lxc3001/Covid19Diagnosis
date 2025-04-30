import torch
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import recall_score
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from classificationDataset import ClassificationDataset, label_to_idx, process_record
from baseline import run_baseline, load_pretrained_encoders

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch.nn as nn
from torchvision.models import resnet50
from transformers import AutoModel, BertModel


def compute_recall(rank_matrix, k=1):
    """Recall@k"""
    correct = 0
    for i in range(rank_matrix.shape[0]):
        if i in rank_matrix[i, :k]:
            correct += 1
    return correct / rank_matrix.shape[0]


@torch.no_grad()
def evaluate_retrieval(image_encoder, text_encoder, dataloader, is_baseline=False, proj_dim=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_encoder.eval()
    text_encoder.eval()

    image_embeddings = []
    text_embeddings = []

    for batch in tqdm(dataloader, desc="Encoding"):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        img_feat = image_encoder(images)
        txt_feat = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(txt_feat, 'last_hidden_state'):
            txt_feat = txt_feat.last_hidden_state[:, 0, :]

        image_embeddings.append(img_feat)
        text_embeddings.append(txt_feat)

    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    # Apply projection to baseline encoders
    if is_baseline:
        image_proj = nn.Linear(image_embeddings.shape[1], proj_dim).to(device)
        text_proj = nn.Linear(text_embeddings.shape[1], proj_dim).to(device)
        image_embeddings = image_proj(image_embeddings)
        text_embeddings = text_proj(text_embeddings)

    # Normalize
    image_embeddings = F.normalize(image_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)

    # Cosine similarity
    sim_matrix = image_embeddings @ text_embeddings.T

    # Ranking
    ranks_i2t = torch.argsort(-sim_matrix, dim=1).cpu().numpy()
    ranks_t2i = torch.argsort(-sim_matrix.T, dim=1).cpu().numpy()

    # Compute recalls
    def compute_recall(rank_matrix, k):
        return np.mean([i in rank_matrix[i, :k] for i in range(len(rank_matrix))])

    result = {
        "i2t": {f"r{k}": compute_recall(ranks_i2t, k) for k in [1, 5, 10]},
        "t2i": {f"r{k}": compute_recall(ranks_t2i, k) for k in [1, 5, 10]},
    }

    print(f"\nImage-to-Text Retrieval:")
    for k in [1, 5, 10]:
        print(f"Recall@{k}: {result['i2t'][f'r{k}']:.4f}")

    print(f"\nText-to-Image Retrieval:")
    for k in [1, 5, 10]:
        print(f"Recall@{k}: {result['t2i'][f'r{k}']:.4f}")

    return result


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
    log_file = open("experiment_retrieval_log.txt", "w", encoding="utf-8")
    tee = Tee(sys.__stdout__, log_file)
    sys.stdout = tee

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        for model_info in models_info:
            model_type = model_info["model_type"]
            print(f"\n========== Running {model_type} ==========")

            tokenizer_name = model_info["tokenizer"]
            dataset = ClassificationDataset(dataframe, img_root, tokenizer_name=tokenizer_name)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

            if model_info.get("baseline", False):
                image_encoder, text_encoder = run_baseline(model_type)
            else:
                image_encoder, text_encoder = load_pretrained_encoders(model_type, model_info["path"])

            image_encoder = image_encoder.to(device)
            text_encoder = text_encoder.to(device)

            result = evaluate_retrieval(image_encoder, text_encoder, dataloader, is_baseline=model_info.get("baseline", False))
            for direction in ["i2t", "t2i"]:
                            all_results.append({
                                "Round": round_idx + 1,
                                "Model": model_type,
                                "Direction": direction,
                                "Recall@1": result[direction]["r1"],
                                "Recall@5": result[direction]["r5"],
                                "Recall@10": result[direction]["r10"]
                            })

    # Save all rounds results
    pd.DataFrame(all_results).to_csv("retrieval_results.csv", index=False)
    print("\nSaved all experiment results to retrieval_results.csv")

    sys.stdout = sys.__stdout__
    log_file.close()