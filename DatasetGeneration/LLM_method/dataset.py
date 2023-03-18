import torch
from torch.utils.data import Dataset
import json
import re

class DocumentStore(Dataset):

    def __init__(self, file_path):
        self.file_path = file_path

        self.documents = []
        self.original_documents = []
        pattern = r'[^\x00-\x7F]'
        with open(self.file_path, "r") as json_file:
            dataset = json.load(json_file)

            for document in dataset:
                self.documents.append(re.sub(pattern, '', document["text"][:2000]))
                self.original_documents.append(document["text"])
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        return {
            "document": self.documents[idx],
            "original_document": self.original_documents[idx]
        }