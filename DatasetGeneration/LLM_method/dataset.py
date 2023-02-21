import torch
from torch.utils.data import Dataset
import json

class DocumentStore(Dataset):

    def __init__(self, file_path):
        self.file_path = file_path

        self.documents = []
        with open(self.file_path, "r") as json_file:
            dataset = json.load(json_file)

            for document in dataset:
                self.documents.append(document["text"])
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        return {
            "document": self.documents[idx]
        }