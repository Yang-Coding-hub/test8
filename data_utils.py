import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from configs import Config

class MultiTurnAdversarialDataset(Dataset):
    def __init__(self, histories, labels, tokenizer):
        self.histories = histories 
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self): return len(self.histories)

    def __getitem__(self, idx):
        session = self.histories[idx]
        turn_ids, turn_masks = [], []
        for turn in session:
            enc = self.tokenizer(turn, truncation=True, max_length=Config.MAX_LENGTH, 
                                 padding="max_length", return_tensors="pt")
            turn_ids.append(enc["input_ids"].squeeze(0))
            turn_masks.append(enc["attention_mask"].squeeze(0))
        return torch.stack(turn_ids), torch.stack(turn_masks), len(session), self.labels[idx]

def collate_fn(batch):
    ids, masks, lengths, labels = zip(*batch)
    return (nn.utils.rnn.pad_sequence(ids, batch_first=True),
            nn.utils.rnn.pad_sequence(masks, batch_first=True),
            torch.tensor(lengths), torch.tensor(labels))

def get_dataloader(dataset, batch_size=Config.BATCH_SIZE):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)