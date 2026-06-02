import argparse
import json
import pickle

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import ShakespeareDataset
from model_lstm import LSTMModel
from model_transformer import TransformerModel

def train(model_name):

    dataset = ShakespeareDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    with open("models/char_to_int.pkl", "rb") as f:
        vocab_size = len(pickle.load(f))

    if model_name == "lstm":
        model = LSTMModel(vocab_size=vocab_size)

    else:
        model = TransformerModel(vocab_size=vocab_size)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    losses = []

    epochs = 1

    model.train()

    for epoch in range(epochs):

        running_loss = 0

        for batch_idx, (x, y) in enumerate(dataloader):

            optimizer.zero_grad()

            if model_name == "lstm":

                hidden = model.init_hidden(
                    x.size(0)
                )

                outputs, hidden = model(
                    x,
                    hidden
                )

            else:

                outputs = model(x)

            loss = criterion(
                outputs.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 200 == 0:

                print(
                    f"Epoch {epoch+1} "
                    f"Batch {batch_idx} "
                    f"Loss {loss.item():.4f}"
                )

            losses.append(loss.item())

            if batch_idx >= 5:
                break

        avg_loss = running_loss / (batch_idx + 1)

        print(
            f"Epoch {epoch+1} "
            f"Average Loss: {avg_loss:.4f}"
        )

    torch.save(
        model.state_dict(),
        f"models/{model_name}_model.pth"
    )

    with open(
        f"models/{model_name}_loss.json",
        "w"
    ) as f:

        json.dump(losses, f)

    print("Training complete")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["lstm", "transformer"],
        required=True
    )

    args = parser.parse_args()

    train(args.model)