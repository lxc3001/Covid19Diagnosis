import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import BertModel
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer
import os
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

# ----------------- Model Definition -----------------
class ConVIRTImageEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(ConVIRTImageEncoder, self).__init__()
        backbone = resnet50(pretrained=True)
        self.visual_backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        x = self.visual_backbone(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.projection(x)
        return F.normalize(x, dim=1)


class ConVIRTTextEncoder(nn.Module):
    def __init__(self, output_dim=128, pretrained_name="bert-base-uncased"):
        super(ConVIRTTextEncoder, self).__init__()
        self.text_encoder = BertModel.from_pretrained(pretrained_name)
        self.projection = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_token)
        return F.normalize(projected, dim=1)

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        labels = torch.arange(image_features.size(0)).to(image_features.device)
        loss_i = nn.CrossEntropyLoss()(logits, labels)
        loss_t = nn.CrossEntropyLoss()(logits.T, labels)
        return (loss_i + loss_t) / 2

# ----------------- Dataloader-----------------
# Image and text augmentation
image_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def random_sentence_sampling(text):
    sentences = text.split('. ')
    if len(sentences) == 0:
        return text
    return random.choice(sentences)


# Convirt Dataset
class ConVIRTDataset(Dataset):
    def __init__(self, dataframe, image_root, tokenizer, transform):
        self.dataframe = dataframe
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_root, row['filename'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        text = random_sentence_sampling(row['clinical_notes'])
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return image, input_ids, attention_mask
    

# ----------------- Training Phase -----------------
def pretrain_convirt(image_encoder, text_encoder, train_loader, num_epochs=20, learning_rate=1e-4, save_dir='./convirt_checkpoints'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_encoder.to(device)
    text_encoder.to(device)

    criterion = InfoNCELoss().to(device)
    optimizer = optim.Adam(
        list(image_encoder.parameters()) + list(text_encoder.parameters()),
        lr=learning_rate
    )

    # Save directory
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        image_encoder.train()
        text_encoder.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for images, input_ids, attention_mask in pbar:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            image_features = image_encoder(images)
            text_features = text_encoder(input_ids, attention_mask)

            loss = criterion(image_features, text_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'image_encoder_state_dict': image_encoder.state_dict(),
            'text_encoder_state_dict': text_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        checkpoint_path = os.path.join(save_dir, f"convirt_epoch_{epoch + 1}.pth")
        torch.save(checkpoint, checkpoint_path)


# ----------------- Main Function -----------------
if __name__ == "__main__":
    csv_path = './data/longitudinal_training_data.csv'
    img_root = './covid-chestxray-dataset-master/data/new_images'

    dataframe = pd.read_csv(csv_path).reset_index(drop=True)

    tokenizer_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    dataset = ConVIRTDataset(dataframe, img_root, tokenizer=tokenizer, transform=image_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, prefetch_factor=2, persistent_workers=True )

    image_encoder = ConVIRTImageEncoder(output_dim=128)
    text_encoder = ConVIRTTextEncoder(output_dim=128)

    pretrain_convirt(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        train_loader=dataloader, 
        num_epochs=20,
        learning_rate=1e-4,
        save_dir='./convirt_checkpoints'
    )

