from src.model_transformer import TransformerModel
import torch

model = TransformerModel(vocab_size=65)

x = torch.randint(
    0,
    65,
    (4, 100)
)

output = model(x)

print(output.shape)