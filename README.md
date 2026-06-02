# Character-Level Text Generation with LSTM and Transformer

## Overview

This project implements and compares two fundamental sequence modeling architectures for character-level text generation using PyTorch:

- Long Short-Term Memory (LSTM)
- Transformer Encoder

Both models are trained on the Tiny Shakespeare dataset and evaluated based on training loss, perplexity, and generated text quality.

The goal is to understand how different neural architectures learn language patterns and generate text one character at a time.

---

## Features

- Character-level text generation
- Custom data preprocessing pipeline
- LSTM implementation using PyTorch
- Mini Transformer implementation using PyTorch
- Temperature-based text sampling
- Training loss visualization
- Quantitative and qualitative model comparison
- Dockerized environment for reproducibility

---

## Project Structure

```text
.
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── README.md
├── requirements.txt
│
├── input/
│   └── shakespeare.txt
│
├── models/
│   ├── char_to_int.pkl
│   ├── int_to_char.pkl
│   ├── data.json
│   ├── lstm_model.pth
│   ├── lstm_loss.json
│   ├── transformer_model.pth
│   └── transformer_loss.json
│
├── results/
│   ├── loss_curves.png
│   ├── generated_samples.json
│   └── comparison_report.md
│
└── src/
    ├── __init__.py
    ├── prepare_data.py
    ├── dataset.py
    ├── model_lstm.py
    ├── model_transformer.py
    ├── train.py
    ├── generate.py
    └── plot_losses.py
```

---

## Dataset

The project uses the Tiny Shakespeare dataset.

Dataset Characteristics:

- Size: ~1.1 MB
- Vocabulary Size: 65 unique characters
- Character-level encoding
- Suitable for CPU training

---

## Data Preparation

The preprocessing pipeline:

1. Reads the text corpus.
2. Extracts unique characters.
3. Creates:
   - `char_to_int`
   - `int_to_char`
4. Encodes the entire dataset.
5. Saves mappings for training and generation.

Run:

```bash
docker-compose run --rm app python src/prepare_data.py
```

---

## Model Architectures

### LSTM Model

Architecture:

```text
Character IDs
      ↓
Embedding Layer
      ↓
LSTM
      ↓
Linear Layer
      ↓
Vocabulary Probabilities
```

Key Components:

- Embedding Layer
- Multi-layer LSTM
- Fully Connected Output Layer

---

### Transformer Model

Architecture:

```text
Character IDs
      ↓
Embedding Layer
      ↓
Positional Encoding
      ↓
Transformer Encoder
      ↓
Linear Layer
      ↓
Vocabulary Probabilities
```

Key Components:

- Token Embeddings
- Positional Encoding
- Multi-Head Self Attention
- Feed Forward Layers
- Layer Normalization

---

## Installation

### Clone Repository

```bash
git clone https://github.com/ashifa-1/character-text-generation
cd character-text-generation
```

### Build Docker Image

```bash
docker-compose build
```

---

## Training

### Train LSTM

```bash
docker-compose run --rm app python src/train.py --model lstm
```

### Train Transformer

```bash
docker-compose run --rm app python src/train.py --model transformer
```

Trained models are saved in:

```text
models/
```

---

## Text Generation

### Generate with LSTM

```bash
docker-compose run --rm app python src/generate.py \
--model lstm \
--model_path models/lstm_model.pth \
--seed_text "To be" \
--temperature 1.0
```

### Generate with Transformer

```bash
docker-compose run --rm app python src/generate.py \
--model transformer \
--model_path models/transformer_model.pth \
--seed_text "To be" \
--temperature 1.0
```

---

## Temperature Sampling

Temperature controls randomness during generation.

| Temperature | Behavior |
|------------|------------|
| 0.5 | Conservative and coherent |
| 1.0 | Balanced |
| 1.5 | Creative but less stable |

Formula:

```text
softmax(logits / temperature)
```

---

## Results

### Training Loss Comparison

Training loss curves are available in:

```text
results/loss_curves.png
```

The LSTM showed gradual convergence.

The Transformer achieved very low training loss rapidly, indicating strong memorization of the training corpus.

---

### Generated Samples

Generated text samples are stored in:

```text
results/generated_samples.json
```

Samples were generated at:

- Temperature 0.5
- Temperature 1.0
- Temperature 1.5

for both models.

---

### Perplexity Comparison

| Model | Final Loss | Approximate Perplexity |
|---------|---------|---------|
| LSTM | 1.4753 | 4.37 |
| Transformer | 0.0326 | 1.03 |

Perplexity was estimated using:

```text
Perplexity = exp(loss)
```

---

## Observations

### LSTM

Advantages:

- Produced coherent Shakespeare-like text
- Generated meaningful words and dialogue structure
- Responded well to temperature changes

Limitations:

- Higher training loss
- Slower convergence

---

### Transformer

Advantages:

- Extremely low training loss
- Fast convergence

Limitations:

- Repetitive generation patterns
- Evidence of mode collapse
- Poor diversity during sampling

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Docker
- Docker Compose

---

## Future Improvements

- Add causal masking to Transformer attention
- Use train/validation/test split
- Compute exact perplexity on a held-out test set
- Add checkpointing and early stopping
- Experiment with deeper architectures
- Support GPU acceleration

---

## Conclusion

This project demonstrates the implementation and comparison of two foundational sequence models for character-level text generation.

While the Transformer achieved significantly lower training loss, the LSTM produced higher-quality generated text. The results highlight an important lesson in generative AI: lower training loss does not always imply better generation quality.

---