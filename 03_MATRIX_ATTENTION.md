# The Matrix Attention Mechanism

This code demonstrates how attention works at its most fundamental level - as a matrix operation where each word "looks at" all other words in the sequence.

## Key Steps:

### 1. Compute attention scores (line 15):
```ruby
scores = Torch.matmul(x, x.t)  # x * x^T
```
This computes dot products between every pair of word vectors. The result is a similarity matrix where `scores[i,j]` represents how similar word `i` is to word `j`.

### 2. Normalize to probabilities (line 18):
```ruby
attention_weights = Torch.softmax(scores, dim: -1)
```
Softmax converts raw scores into probabilities that sum to 1 for each row. This determines how much each word should "pay attention to" every other word.

### 3. Apply attention (line 21):
```ruby
output = Torch.matmul(attention_weights, x)
```
This creates new representations where each word is a weighted combination of all words, with weights determined by the attention scores.

## The Example:

- **Word 1**: `[1.0, 0.0]` - purely feature 1
- **Word 2**: `[0.0, 1.0]` - purely feature 2  
- **Word 3**: `[0.5, 0.5]` - mix of both features

Word 3 will have high similarity to both words 1 and 2, so its output will blend information from all positions. This is how context flows through the model - each word's representation gets enriched with information from related words.

## Key Insight:

As the README mentions, this is essentially a database query where each word queries all others and aggregates their values based on similarity. The attention mechanism allows the model to dynamically decide which words are relevant to each other, creating context-aware representations.
