import argparse
import pickle

import torch
import torch.nn.functional as F

from model_lstm import LSTMModel
from model_transformer import TransformerModel


def load_mappings():

    with open("models/char_to_int.pkl", "rb") as f:
        char_to_int = pickle.load(f)

    with open("models/int_to_char.pkl", "rb") as f:
        int_to_char = pickle.load(f)

    return char_to_int, int_to_char


def generate_text(
    model,
    seed_text,
    char_to_int,
    int_to_char,
    temperature=1.0,
    length=200,
    model_type="lstm"
):

    model.eval()

    generated = seed_text

    with torch.no_grad():

        for _ in range(length):

            encoded = [
                char_to_int[c]
                for c in generated[-100:]
                if c in char_to_int
            ]

            x = torch.tensor(
                [encoded],
                dtype=torch.long
            )

            if model_type == "lstm":

                hidden = model.init_hidden(1)

                output, hidden = model(
                    x,
                    hidden
                )

            else:

                output = model(x)

            logits = output[0, -1]

            logits = logits / temperature

            probs = F.softmax(
                logits,
                dim=0
            )

            next_idx = torch.multinomial(
                probs,
                1
            ).item()

            next_char = int_to_char[next_idx]

            generated += next_char

    return generated


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["lstm", "transformer"],
        required=True
    )

    parser.add_argument(
        "--model_path",
        required=True
    )

    parser.add_argument(
        "--seed_text",
        required=True
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0
    )

    args = parser.parse_args()

    char_to_int, int_to_char = load_mappings()

    vocab_size = len(char_to_int)

    if args.model == "lstm":

        model = LSTMModel(
            vocab_size=vocab_size
        )

    else:

        model = TransformerModel(
            vocab_size=vocab_size
        )

    model.load_state_dict(
        torch.load(
            args.model_path,
            map_location="cpu"
        )
    )

    text = generate_text(
        model=model,
        seed_text=args.seed_text,
        char_to_int=char_to_int,
        int_to_char=int_to_char,
        temperature=args.temperature,
        model_type=args.model
    )

    print("\nGenerated Text:\n")
    print(text)