import pickle
import torch
from torch.utils.data import Dataset
from pathlib import Path


class ShakespeareDataset(Dataset):

    def __init__(
        self,
        text_path="input/shakespeare.txt",
        seq_length=100
    ):
        self.seq_length = seq_length

        text = Path(text_path).read_text(
            encoding="utf-8"
        )

        with open(
            "models/char_to_int.pkl",
            "rb"
        ) as f:
            self.char_to_int = pickle.load(f)

        self.encoded = [
            self.char_to_int[c]
            for c in text
        ]

    def __len__(self):
        return len(self.encoded) - self.seq_length

    def __getitem__(self, idx):

        x = self.encoded[
            idx : idx + self.seq_length
        ]

        y = self.encoded[
            idx + 1 : idx + self.seq_length + 1
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )