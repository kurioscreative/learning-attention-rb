# Level 4: Scaled Dot-Product Attention Explained

## Overview
This file introduces the **Scaled Dot-Product Attention** mechanism - the fundamental building block of the Transformer architecture. It implements the famous Query-Key-Value (Q-K-V) attention mechanism that allows each element in a sequence to attend to all other elements.

## The Core Concept

### The Q-K-V Metaphor
Think of it like a search engine:
- **Query (Q)**: "What am I looking for?" - The current word asking questions
- **Key (K)**: "What can I match against?" - The searchable attributes of all words
- **Value (V)**: "What information do I actually want?" - The actual content to retrieve

### The Database Analogy
As mentioned in the README, attention is like a database query:
```sql
SELECT weighted_avg(values) 
FROM words 
WHERE similarity(query, key) > threshold
```

## Code Walkthrough

### 1. Computing Attention Scores (Lines 13-19)
```ruby
d_k = query.size(-1)  # Get dimension of keys
scores = Torch.matmul(query, key.transpose(-2, -1))
scores /= Math.sqrt(d_k)
```

- **Matrix multiplication**: `query @ key.T` computes how similar each query is to each key
- **Scaling**: Dividing by `d_k` prevents the dot products from growing too large
  - Without scaling, large values would cause softmax to output near-binary distributions
  - This maintains gradient flow during training

### 2. Converting to Probabilities (Line 22)
```ruby
attention_weights = Torch.softmax(scores, dim: -1)
```

- Softmax converts raw scores into probabilities that sum to 1
- Each query position gets a probability distribution over all key positions
- Higher scores = higher attention weights

### 3. Applying Attention (Line 25)
```ruby
output = Torch.matmul(attention_weights, value)
```

- This is the weighted average of values
- Each output position is a blend of all value vectors, weighted by attention

## Why Scaling Matters

The scaling factor `1/d_k` is crucial:
- As dimension `d_k` increases, dot products grow larger
- Large dot products ’ extreme softmax outputs ’ vanishing gradients
- Scaling keeps the variance stable regardless of dimension size

## Example Shapes

For the test case with shape `[2, 3, 4]`:
- **Batch size**: 2 (processing 2 sequences in parallel)
- **Sequence length**: 3 (each sequence has 3 positions)
- **Dimension**: 4 (each position represented by 4 numbers)

The attention computation:
1. `scores`: [2, 3, 3] - Each of 3 positions attends to all 3 positions
2. `attention_weights`: [2, 3, 3] - Normalized to probabilities
3. `output`: [2, 3, 4] - Same shape as input, but enriched with context

## Key Insights

1. **Self-Attention**: When Q, K, V come from the same source, it's self-attention
2. **Parallelization**: All positions can be computed simultaneously (unlike RNNs)
3. **Information Flow**: Each output contains information from all inputs
4. **Permutation Invariance**: Without position encoding, order doesn't matter

## Connection to Previous Levels

- **Level 1-2**: Basic attention between individual elements
- **Level 3**: Matrix form for efficiency
- **Level 4**: Adds scaling and the Q-K-V abstraction

This scaled dot-product attention is used multiple times in multi-head attention (next level), where different heads learn to attend to different types of relationships.