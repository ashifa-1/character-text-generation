import json
import pickle
from pathlib import Path

DATA_PATH = "input/shakespeare.txt"


def prepare_data():
    print("Starting preprocessing...")

    text = Path(DATA_PATH).read_text(encoding="utf-8")

    print("Dataset loaded")

    chars = sorted(list(set(text)))

    vocab_size = len(chars)

    print(f"Vocabulary Size: {vocab_size}")
    print(f"Dataset Length: {len(text)}")

    char_to_int = {ch: idx for idx, ch in enumerate(chars)}
    int_to_char = {idx: ch for idx, ch in enumerate(chars)}

    encoded = [char_to_int[c] for c in text]

    Path("models").mkdir(exist_ok=True)

    with open("models/char_to_int.pkl", "wb") as f:
        pickle.dump(char_to_int, f)

    with open("models/int_to_char.pkl", "wb") as f:
        pickle.dump(int_to_char, f)

    with open("models/data.json", "w") as f:
        json.dump(
            {
                "vocab_size": vocab_size,
                "encoded_length": len(encoded)
            },
            f,
            indent=4
        )

    print("Mappings saved successfully")


if __name__ == "__main__":
    print("File executed directly")
    prepare_data()