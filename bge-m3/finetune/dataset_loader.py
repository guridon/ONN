import json
import torch
from torch.utils.data import Dataset, DataLoader

class CodeSwitchDataset(Dataset):
    def __init__(self, queries, tokenizer, max_length=128):
        """
        Args:
            queries (list): [{"EtoK": ..., "KtoE": ..., "English": ..., "Korean": ...}, ...]
        """
        self.queries = queries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query_group = self.queries[idx]
        encodings = {
            k: self.tokenizer(v, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
            for k, v in query_group.items()
        }
        return encodings

def load_code_switch_dataset(file_path, tokenizer):
    """
    Args:
        file_path (str): 데이터셋 파일 경로 (JSON 형식)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    dataset = CodeSwitchDataset(queries, tokenizer)
    return DataLoader(dataset, 
                      batch_size=16, 
                      shuffle=True, 
                      num_workers=4,        
                    pin_memory=True)
