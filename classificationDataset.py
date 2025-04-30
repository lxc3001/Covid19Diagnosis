import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms


# ----------------- Label Projection -----------------
category_map = {
    'COVID-19': 'Viral', 'SARS': 'Viral', 'MERS-CoV': 'Viral', 'Influenza': 'Viral',
    'Varicella': 'Viral', 'Herpes': 'Viral', 'H1N1': 'Viral',
    'Streptococcus': 'Bacterial', 'E.coli': 'Bacterial', 'Klebsiella': 'Bacterial',
    'Nocardia': 'Bacterial', 'Legionella': 'Bacterial', 'Chlamydophila': 'Bacterial',
    'Staphylococcus': 'Bacterial', 'MRSA': 'Bacterial', 'Mycoplasma': 'Bacterial',
    'Tuberculosis': 'Bacterial',
    'Pneumocystis': 'Fungal', 'Aspergillosis': 'Fungal',
    'Lipoid': 'Others', 'Aspiration': 'Others',
    'todo': 'Unknown', 'Unknown': 'Unknown',
    'No Finding': 'No Findings'
}

label_to_idx = {'Viral': 0, 'Bacterial': 1, 'Fungal': 2, 'Others': 3, 'Unknown': 4, 'Pneumonia': 5, 'No Findings': 6}

def merge_label(old_label):
    return category_map.get(old_label.strip(), old_label.strip())

def process_record(record):
    parts = record.split('/')
    merged_labels = [merge_label(part) for part in parts]
    if len(set(merged_labels)) == 1:
        return merged_labels[0]
    priority = {'Viral': 1, 'Bacterial': 2, 'Fungal': 3, 'Others': 4, 'Unknown': 5, 'Pneumonia': 6, 'No Findings': 7}
    prioritized = sorted([(priority.get(label, 100), label) for label in merged_labels], key=lambda x: x[0])
    return prioritized[0][1]

# ----------------- Dataset -----------------
class ClassificationDataset(Dataset):
    def __init__(self, dataframe, img_dir, tokenizer_name='bert-base-uncased', max_length=512, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        text = str(row['clinical_notes'])
        encoded_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = encoded_text['input_ids'].squeeze(0)
        attention_mask = encoded_text['attention_mask'].squeeze(0)
        label = torch.tensor(row['final_label'], dtype=torch.long)

        return {'image': image, 'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}