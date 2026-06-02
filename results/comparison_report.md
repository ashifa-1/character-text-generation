# Character-Level Text Generation Comparison

### Perplexity Comparison

| Model | Final Loss | Approximate Perplexity |
|---------|---------|---------|
| LSTM | 1.4753 | 4.37 |
| Transformer | 0.0326 | 1.03 |

Perplexity was estimated using:

perplexity = exp(loss)

Lower values indicate higher confidence in next-character prediction.

---

### Qualitative Analysis

The LSTM generated coherent Shakespeare-like text with recognizable words, punctuation, dialogue formatting, and character names. The generated samples demonstrated the ability to learn sentence structure and stylistic patterns from the training corpus.

The Transformer achieved a substantially lower training loss but produced repetitive text during generation. Regardless of temperature settings, it repeatedly generated sequences such as:

"To beeeeeeeeeeeeeeeee..."

This indicates mode collapse or severe overfitting, where the model became extremely confident in predicting a single character repeatedly.

As a result, despite achieving better quantitative metrics, the Transformer produced lower-quality text samples than the LSTM.

---

### Temperature Analysis

Temperature had a significant impact on the LSTM model.

- Temperature 0.5 produced more coherent and conservative text.
- Temperature 1.0 balanced coherence and creativity.
- Temperature 1.5 produced more diverse but less grammatical text.

For the Transformer model, changing temperature had little effect because the model had already collapsed to a highly confident repetitive prediction pattern.

---

### Conclusion

The LSTM provided the best balance between coherence and creativity for this dataset.

Transformer achieved significantly lower training loss and perplexity.
However, generated samples revealed severe repetition and mode collapse,
suggesting that low training loss alone does not guarantee high-quality
text generation.
