# Understanding 01_simple_attention.rb

## Overview

This file implements the absolute simplest version of "attention" - the core mechanism behind Transformers. It demonstrates the fundamental concept: **looking at all elements and deciding which ones matter most**.

## The Code Breakdown

### The SimpleAttention Class

```ruby
class SimpleAttention
  def forward(words)
    # Step 1: Calculate importance scores
    importance = Torch.softmax(words, dim:0)
    
    # Step 2: Create weighted average
    output = (words * importance).sum
    
    output
  end
end
```

### What's Happening Step by Step

1. **Input**: A tensor of values `[0.1, 0.5, 0.3, 0.9]`
   - Think of these as simplified "word representations"
   - In real transformers, these would be high-dimensional vectors

2. **Softmax Operation**: `Torch.softmax(words, dim:0)`
   - Converts raw values into probabilities that sum to 1
   - Higher input values get exponentially higher probabilities
   - For input `[0.1, 0.5, 0.3, 0.9]`:
     - Softmax gives approximately `[0.16, 0.27, 0.21, 0.36]`
     - Notice how 0.9 (highest) gets 36% of the "attention"

3. **Weighted Average**: `(words * importance).sum`
   - Multiplies each value by its importance score
   - Sums everything up to get a single output value
   - This is essentially asking: "What's the weighted average of all words based on their importance?"

## The Key Insight

This simple mechanism demonstrates the core idea of attention:
- **All elements contribute** to the output (unlike RNNs that process sequentially)
- **Contribution is weighted** by importance/relevance
- **Importance is learned** from the data itself

## Connection to Full Transformers

In real Transformers:
- Instead of using values directly as importance scores, we compute Query, Key, and Value transformations
- Instead of single numbers, we work with high-dimensional embeddings
- Instead of one attention operation, we stack many in parallel (multi-head) and in sequence (layers)

But the core principle remains: **weighted aggregation based on learned importance**.

## Why This Matters

This simple attention already solves a key problem:
- **Parallel processing**: All words are processed simultaneously
- **Dynamic weighting**: The model decides which words to focus on
- **Differentiable**: Everything uses standard operations that can be backpropagated

## Next Steps

The next file (`02_attention_between_two_things.rb`) will introduce the concept of Query and Key - allowing one word to "ask" about another word's relevance, rather than words just declaring their own importance.