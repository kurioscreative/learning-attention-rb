# Understanding 02_attention_between_two_things.rb

## Overview

This file introduces the crucial concept of **Query-Key attention** - where one element (query) looks for relevant information in other elements (keys). This is a major step toward real Transformer attention.

## The Evolution from Level 1

In Level 1, words decided their own importance. Now we introduce:
- **Query**: "What am I looking for?"
- **Keys**: "What options are available to look at?"

This models how attention really works in language: one word asking "which other words are relevant to me?"

## The Code Breakdown

### The PairwiseAttention Class

```ruby
class PairwiseAttention
  def forward(query, keys)
    # Step 1: Compute similarity scores
    scores = query * keys
    
    # Step 2: Convert to probabilities
    attention_weights = Torch.softmax(scores, dim: 0)
    
    attention_weights
  end
end
```

### What's Happening Step by Step

1. **Input Setup**:
   - Query: `0.5` (a single value representing "what I'm looking for")
   - Keys: `[0.1, 0.5, 0.3, 0.9]` (multiple values representing available words)

2. **Similarity Calculation**: `query * keys`
   - Element-wise multiplication: `0.5 * [0.1, 0.5, 0.3, 0.9]`
   - Results in: `[0.05, 0.25, 0.15, 0.45]`
   - Higher scores mean higher similarity between query and key

3. **Softmax Normalization**:
   - Converts raw scores to probabilities
   - The key at position 1 (value 0.5) gets high attention because it matches the query
   - The key at position 3 (value 0.9) gets the highest attention due to larger product

## The Key Insights

### 1. Similarity as Multiplication
Simple multiplication acts as a similarity metric:
- Same sign + large magnitude = high similarity
- Opposite signs = negative attention (pushed down by softmax)
- Zero in either = no attention

### 2. Query-Key Separation
This separation models real questions:
- Query = "I'm a verb looking for my subject"
- Keys = All words that could be subjects
- Attention weights = Probability each word is the subject

### 3. Foundation for Scaled Dot-Product
This simple multiplication will evolve into:
```
attention_scores = (Q @ K.T) / sqrt(d_k)
```
But the principle remains: measuring similarity between queries and keys.

## Connection to Database Queries

As the README mentions, this is like a soft database query:

```sql
-- Hard database query (returns specific rows)
SELECT * FROM words WHERE similarity = 0.5

-- Soft attention (returns weighted combination)
SELECT weighted_avg(*) FROM words 
WEIGHTED BY similarity_to(0.5)
```

## Why This Architecture Matters

1. **Flexible Relationships**: Any word can attend to any other word
2. **Learned Similarity**: The model learns what "similar" means for the task
3. **Parallelizable**: All similarities computed simultaneously
4. **Differentiable**: Gradients flow through multiplication and softmax

## Limitations of This Version

1. **Fixed Similarity Metric**: Just multiplication - real Transformers learn projections
2. **No Values**: We return attention weights, not weighted content
3. **Single Dimension**: Real attention uses high-dimensional vectors
4. **One Query**: Full attention has every word query every other word

## Next Steps

The next file (`03_matrix_attention.rb`) will:
- Introduce the Value component (Query, Key, **Value**)
- Use matrix operations for efficiency
- Allow every word to query every other word simultaneously

This pairwise attention is the conceptual bridge between simple weighted averaging and the full attention mechanism used in Transformers.