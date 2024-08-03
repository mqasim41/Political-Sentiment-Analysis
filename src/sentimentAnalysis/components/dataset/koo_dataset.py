from torch.utils.data import Dataset
class KooDataset(Dataset):
    def __init__(self, inputs, targets, attention_mask):
        self.inputs = inputs
        self.targets = targets
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.targets[idx]
        }