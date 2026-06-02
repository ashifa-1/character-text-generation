import json
import matplotlib.pyplot as plt

with open("models/lstm_loss.json") as f:
    lstm_loss = json.load(f)

with open("models/transformer_loss.json") as f:
    transformer_loss = json.load(f)

plt.figure(figsize=(10, 6))

plt.plot(lstm_loss, label="LSTM")
plt.plot(transformer_loss, label="Transformer")

plt.title("Training Loss Comparison")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.legend()

plt.savefig("results/loss_curves.png")
print("loss_curves.png created")