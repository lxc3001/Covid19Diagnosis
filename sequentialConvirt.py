import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import monai.transforms as mt

# ----------------- Dataloader-----------------
# Image augmentation 
image_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
    transforms.RandomHorizontalFlip(),                 
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class TemporalDataset(Dataset):
    def __init__(self, data_frame, img_dir, max_length=128, img_transform=None):
        self.data_frame = data_frame
        self.img_dir = img_dir
        self.max_length = max_length
        self.img_transform = img_transform if img_transform is not None else transforms.ToTensor()
        self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        
        # Group by patient ID
        self.patient_groups = self.data_frame.groupby('patientid')
        self.patient_ids = list(self.patient_groups.groups.keys())
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        # Sorted by offset to ensure temporal order
        patient_id = self.patient_ids[idx]
        patient_data = self.patient_groups.get_group(patient_id).sort_values('offset')
        
        images = []
        input_ids_list = []
        attention_mask_list = []
        
        for _, row in patient_data.iterrows():
            # Image
            img_name = row['filename']
            img_path = os.path.join(self.img_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
            image_aug = self.img_transform(image)
            images.append(image_aug)
            
            # Text
            text = row['clinical_notes_new']
            tokenized = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            input_ids = tokenized['input_ids'].squeeze(0)
            attention_mask = tokenized['attention_mask'].squeeze(0)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        
        # images: [T, 3, H, W]; input_ids: [T, L]; attention_mask: [T, L]
        images = torch.stack(images, dim=0)
        input_ids = torch.stack(input_ids_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)
        length = images.size(0)

        return images, input_ids, attention_mask, length

# Handle patients with different temporal lengths
def collate_fn(batch):
    max_T = max(item[3] for item in batch)
    
    images_list, input_ids_list, attention_mask_list, lengths = [], [], [], []
    for images, input_ids, attention_mask, length in batch:
        lengths.append(length)
        T, C, H, W = images.size()
        # Padding if length less than max length in the batch
        if T < max_T:
            pad_images = torch.zeros((max_T - T, C, H, W), dtype=images.dtype)
            images = torch.cat([images, pad_images], dim=0)
            pad_input_ids = torch.zeros((max_T - T, input_ids.size(1)), dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad_input_ids], dim=0)
            pad_attention_mask = torch.zeros((max_T - T, attention_mask.size(1)), dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, pad_attention_mask], dim=0)
        images_list.append(images)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        
    # Stacking all images, input_ids, and attention_mask
    images_batch = torch.stack(images_list, dim=0)          # [B, max_T, 3, H, W]
    input_ids_batch = torch.stack(input_ids_list, dim=0)      # [B, max_T, L]
    attention_mask_batch = torch.stack(attention_mask_list, dim=0)  # [B, max_T, L]
    lengths = torch.tensor(lengths)                         # [B]
    return images_batch, input_ids_batch, attention_mask_batch, lengths



# Image Encoder
class SequentialImageEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super(SequentialImageEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  
        projected = self.projection(features)       
        projected = F.normalize(projected, p=2, dim=1) 
        return projected

# Text Encoder
class SequentialTextEncoderBERT(nn.Module):
    def __init__(self, output_dim=256, pretrained_model_name="medicalai/ClinicalBERT"):
        super(SequentialTextEncoderBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.projection = nn.Linear(self.bert.config.hidden_size, output_dim)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(pooled)
        projected = F.normalize(projected, p=2, dim=1)
        return projected


# LSTM Temporal Aggregator
class TemporalAggregator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_layers=1):
        super(TemporalAggregator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x, lengths):
        # x: [B, T, D]，lengths: [B]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        return h_n[-1]  # [B, hidden_dim]


# Complete SequentialModel
class TemporalModel(nn.Module):
    def __init__(self, img_output_dim=256, txt_output_dim=256, lstm_hidden_dim=256, pretrained_text_model="medicalai/ClinicalBERT"):
        super(TemporalModel, self).__init__()
        self.image_encoder = SequentialImageEncoder(output_dim=img_output_dim)
        self.text_encoder  = SequentialTextEncoderBERT(output_dim=txt_output_dim, pretrained_model_name=pretrained_text_model)
        self.img_temporal_aggregator = TemporalAggregator(input_dim=img_output_dim, hidden_dim=lstm_hidden_dim)
        self.txt_temporal_aggregator = TemporalAggregator(input_dim=txt_output_dim, hidden_dim=lstm_hidden_dim)
    
    def forward(self, images_seq, input_ids_seq, attention_mask_seq, lengths):
        B, T, _, _, _ = images_seq.size()
        img_feats_seq = []
        txt_feats_seq = []
        
        for t in range(T):
            images_t = images_seq[:, t, :, :, :]      # [B, 3, H, W]
            input_ids_t = input_ids_seq[:, t, :]        # [B, L]
            attention_mask_t = attention_mask_seq[:, t, :]  # [B, L]
            
            img_feats = self.image_encoder(images_t)
            txt_feats = self.text_encoder(input_ids_t, attention_mask_t)
            img_feats_seq.append(img_feats)
            txt_feats_seq.append(txt_feats)
        
        # [B, T, feature_dim]
        img_feats_seq = torch.stack(img_feats_seq, dim=1)
        txt_feats_seq = torch.stack(txt_feats_seq, dim=1)
        
        # Temporal aggregation
        img_agg = self.img_temporal_aggregator(img_feats_seq, lengths)  # [B, lstm_hidden_dim]
        txt_agg = self.txt_temporal_aggregator(txt_feats_seq, lengths)  # [B, lstm_hidden_dim]
        
        return img_agg, txt_agg

# InfoNCE Loss
def info_nce_loss(image_features, text_features, temperature=0.07):
    logits = torch.matmul(image_features, text_features.t()) / temperature  # [B, B]
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    loss = (loss_i + loss_t) / 2.0
    return loss


# ----------------- Training Phase -----------------
def pretrain_temporalModel(model, train_loader, num_epochs=20, learning_rate=1e-4, save_dir='./temporal_checkpoints'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = info_nce_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for images_seq, input_ids_seq, attention_mask_seq, lengths in pbar:
            images_seq = images_seq.to(device)
            input_ids_seq = input_ids_seq.to(device)
            attention_mask_seq = attention_mask_seq.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            img_agg, txt_agg = model(images_seq, input_ids_seq, attention_mask_seq, lengths)
            loss = criterion(img_agg, txt_agg)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        checkpoint_path = os.path.join(save_dir, f"temporal_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, checkpoint_path)


# ----------------- Main Function -----------------
if __name__ == '__main__':
    csv_path = './data/longitudinal_training_data.csv'
    img_root = './covid-chestxray-dataset-master/data/new_images'

    dataframe = pd.read_csv(csv_path).reset_index(drop=True)

    dataset = TemporalDataset(data_frame=dataframe,img_dir=img_root,max_length=128,img_transform=image_transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, prefetch_factor=2, persistent_workers=True, collate_fn=collate_fn)

    model = TemporalModel(img_output_dim=256, txt_output_dim=256, lstm_hidden_dim=256, pretrained_text_model="medicalai/ClinicalBERT")

    pretrain_temporalModel(model=model, train_loader=dataloader, num_epochs=20, learning_rate=1e-4, save_dir='./temporal_checkpoints')
