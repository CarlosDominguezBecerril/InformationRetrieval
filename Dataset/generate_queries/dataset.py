import torch
from torch.utils.data import Dataset
import json

class DPRDataset(Dataset):
    
    def __init__(self, file_path):
        self.file_path = file_path

        self.documents, self.documents_ids = [], []
        with open(self.file_path, "r") as json_file:
            dataset = json.load(json_file)

            for element in dataset:
                for document in element["positive_ctxs"]:
                    self.documents.append(document["text"])
                    self.documents_ids.append(document["passage_id"])

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        passage = self.documents[idx]
        passage_id = self.documents_ids[idx]
       
        outputs = {
            "passage": passage,
            "passage_id": torch.LongTensor([passage_id]),
        }

        return outputs