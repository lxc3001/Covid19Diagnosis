import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
import pandas as pd
import os
import torch.optim as optim
from tqdm import tqdm

# Image Encoder
class MedicalImageEncoder(nn.Module):
    def __init__(self, output_dim=256, freeze_until='layer3'):
        super().__init__()
        resnet = models.resnet50(pretrained=True)

        if freeze_until is not None:
            self.freeze_parameters(resnet, freeze_until)

        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def freeze_parameters(self, model, freeze_until):
        freeze_dict = {
            'layer1': ['conv1', 'bn1', 'layer1'],
            'layer2': ['conv1', 'bn1', 'layer1', 'layer2'],
            'layer3': ['conv1', 'bn1', 'layer1', 'layer2', 'layer3'],
            'layer4': []  # 不冻结
        }
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in freeze_dict[freeze_until]):
                param.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.projection(x), dim=1)

# Text Encoder
class MedicalTextEncoder(nn.Module):
    def __init__(self, output_dim=256, pretrained_name="emilyalsentzer/Bio_ClinicalBERT", freeze_layers=10):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.freeze_bert_layers(freeze_layers)
        
        self.medical_adapter = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, output_dim)
        )
        self._init_weights()

    def freeze_bert_layers(self, num_frozen):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        if hasattr(self.bert, 'encoder'):
            layers = self.bert.encoder.layer
        else:
            raise ValueError("Unsupported model structure")

        for layer in layers[:num_frozen]:
            for param in layer.parameters():
                param.requires_grad = False

    def _init_weights(self):
        nn.init.kaiming_normal_(self.medical_adapter.weight, nonlinearity='relu')
        nn.init.zeros_(self.medical_adapter.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        adapted = self.medical_adapter(outputs.last_hidden_state)
        pooled = self.medical_pooling(adapted, attention_mask)
        return F.normalize(self.projection(pooled), dim=1)

    def medical_pooling(self, hidden_states, mask):
        weights = torch.sigmoid(hidden_states[:, :, 0])
        weights = weights.masked_fill(mask == 0, -1e9)
        weights = F.softmax(weights, dim=1)
        return torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)

# Model
class MultimodalEncoder(nn.Module):
    def __init__(self, image_output_dim=256, text_output_dim=256):
        super().__init__()
        self.image_encoder = MedicalImageEncoder(output_dim=image_output_dim)
        self.text_encoder = MedicalTextEncoder(output_dim=text_output_dim)

    def forward(self, images, input_ids, attention_mask):
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(input_ids, attention_mask)
        return img_feat, txt_feat

# Hierarchical Contrastive Loss
class HierContrastiveLoss(nn.Module):
    def __init__(self, tau_strong=0.07, tau_weak=0.07, tau_neg=0.07, lambda_weak=0.5):
        super().__init__()
        self.tau_strong = tau_strong
        self.tau_weak = tau_weak
        self.tau_neg = tau_neg
        self.lambda_weak = lambda_weak

    def forward(self, v, u, patient_ids):
        batch_size = v.size(0)

        # --------- Inter-patient Loss---------
        sim_matrix = torch.matmul(v, u.t())

        same_patient = patient_ids.unsqueeze(1) == patient_ids.unsqueeze(0)
        strong_mask = torch.eye(batch_size, dtype=torch.bool, device=v.device)
        neg_mask = ~same_patient

        # Strong positive pairs
        sim_matrix_strong = sim_matrix / self.tau_strong
        exp_sim_strong = torch.exp(sim_matrix_strong)
        pos_pairs = (exp_sim_strong * strong_mask).sum(dim=1, keepdim=True)
        
        # Negative pairs
        sim_matrix_neg = sim_matrix / self.tau_neg
        exp_sim_neg = torch.exp(sim_matrix_neg)
        neg_IT = (exp_sim_neg * neg_mask).sum(dim=1, keepdim=True) + 1e-8
        neg_TI = ((exp_sim_neg * neg_mask).sum(dim=0, keepdim=True) + 1e-8).T

        loss_IT = -torch.log(pos_pairs / neg_IT).mean()
        loss_TI = -torch.log(pos_pairs / neg_TI).mean()
        loss_strong = (loss_IT + loss_TI) / 2

        # --------- Intra-patient loss ---------
        sim_matrix_weak = sim_matrix / self.tau_weak
        exp_sim_weak = torch.exp(sim_matrix_weak)

        loss_weak_it, loss_weak_ti = 0.0, 0.0
        count_weak_it, count_weak_ti = 0, 0

        for i in range(batch_size):
            weak_it = same_patient[i] & (~strong_mask[i])
            if weak_it.sum() > 0:
                sim_strong_i = sim_matrix_weak[i, i]
                sim_weak_i = sim_matrix_weak[i][weak_it]
                loss_weak_it += -torch.log(torch.exp(sim_strong_i) / (torch.exp(sim_strong_i) + sim_weak_i.exp().sum()))
                count_weak_it += 1

            weak_ti = same_patient[:, i] & (~strong_mask[:, i])
            if weak_ti.sum() > 0:
                sim_strong_i = sim_matrix_weak[i, i]
                sim_weak_i = sim_matrix_weak[:, i][weak_ti]
                loss_weak_ti += -torch.log(torch.exp(sim_strong_i) / (torch.exp(sim_strong_i) + sim_weak_i.exp().sum()))
                count_weak_ti += 1

        if count_weak_it > 0:
            loss_weak_it /= count_weak_it
        if count_weak_ti > 0:
            loss_weak_ti /= count_weak_ti

        loss_weak = (loss_weak_it + loss_weak_ti) / 2

        # --------- Final Loss ---------
        total_loss = loss_strong + self.lambda_weak * loss_weak
        return total_loss


# Dataset 
image_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class MedicalMultimodalDataset(Dataset):
    def __init__(self, csv_path, img_dir, tokenizer_name="emilyalsentzer/Bio_ClinicalBERT", max_length=128, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.transform = transform

        self.unique_patients = self.data['patientid'].unique()
        self.patient_to_idx = {pid: idx for idx, pid in enumerate(self.unique_patients)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        clinical_note = row['clinical_notes_new']
        patient_id = self.patient_to_idx[row['patientid']]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        encoded_text = self.tokenizer(clinical_note, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoded_text['input_ids'].squeeze(0)
        attention_mask = encoded_text['attention_mask'].squeeze(0)

        return image, input_ids, attention_mask, torch.tensor(patient_id, dtype=torch.long)
    

def pretrain_HierModel(model, train_loader, num_epochs=20, learning_rate=1e-4, save_dir='./hier_checkpoints'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = HierContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for images, input_ids, attention_mask, patient_ids in pbar:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            patient_ids = patient_ids.to(device)

            optimizer.zero_grad()
            img_features, txt_features = model(images, input_ids, attention_mask)
            loss = criterion(img_features, txt_features, patient_ids)

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
        checkpoint_path = os.path.join(save_dir, f"epoch_{epoch + 1}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")


if __name__ == '__main__':
    csv_path = './data/longitudinal_training_data.csv'
    img_root = './covid-chestxray-dataset-master/data/new_images'

    dataset = MedicalMultimodalDataset(csv_path=csv_path, img_dir=img_root, transform=image_transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, prefetch_factor=2, persistent_workers=True)

    model = MultimodalEncoder(image_output_dim=256, text_output_dim=256)

    pretrain_HierModel(model=model, train_loader=data_loader, num_epochs=20, learning_rate=1e-4, save_dir='./hier_checkpoints')
