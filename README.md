# Attention Is All You Need - Ruby Implementation

A step-by-step implementation of the essential Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017), built in Ruby using the torch.rb gem. Each numbered file progressively builds complexity, teaching the concepts through code.

## Prerequisites

### Install LibTorch

The torch.rb gem requires LibTorch (PyTorch's C++ library):

**macOS/Linux with Homebrew:**
```bash
brew install libtorch
```

**Manual installation:**
1. Download LibTorch from https://pytorch.org/get-started/locally/
2. Extract and set environment variable:
   ```bash
   export LIBTORCH=/path/to/libtorch
   bundle config build.torch-rb --with-torch-dir=/path/to/libtorch
   ```

### Install Dependencies
```bash
bundle config build.torch-rb --with-torch-dir=/path/to/libtorch
bundle install
```

For help installing torch.rb and LibTorch, see [ankane/torch.rb](https://github.com/ankane/torch.rb).

## How to Use This Project

Each Ruby file (1_simple_attention.rb through 11_simple_example.rb) builds on the previous one, introducing new concepts:

```bash
# Start with the simplest implementation
ruby 1_simple_attention.rb

# Progress through each file to build understanding
ruby 2_attention_between_two_things.rb
ruby 3_matrix_attention.rb
# ... and so on
```

The files are designed to be read in order, with each one adding a new layer of the Transformer architecture. Corresponding Markdown files include additional commentary.

## The Journey: From Simple to Transformer

Let's build this like we're building any software system - start with the simplest possible working version and incrementally add features. We'll focus on the engineering insights rather than the ML theory.## The Engineering Mental Model

Think of Transformers like this:

### 1. **Attention = Database Query**
```sql
SELECT weighted_avg(values) 
FROM words 
WHERE similarity(query, key) > threshold
```
Each word queries all other words and aggregates their values based on similarity.

### 2. **Multi-Head = Parallel Processing**
Like running multiple regex patterns on the same text - each head looks for different patterns:
- Head 1: Syntactic relationships (subject-verb)
- Head 2: Semantic similarity (cat-dog)
- Head 3: Position patterns (first-last word)

### 3. **Transformer Block = Pipeline Stage**
```ruby
input
  |> attention_layer      # Enrich with context
  |> add_residual        # Preserve original info
  |> normalize           # Stabilize numerics
  |> feedforward        # Process individually
  |> add_residual       # Preserve again
  |> normalize          # Stabilize again
  |> output
```

### 4. **Position Encoding = Array Indices**
Without position encoding, Transformers see sentences like a HashSet instead of an Array. The sinusoidal encoding is just a clever way to give each position a unique "hash" that preserves relative distance information.

## Quick Wins for Understanding

1. **Start with Level 3** - Matrix attention is the core insight. Everything else is optimization.

2. **Ignore the math** - Softmax just means "convert scores to probabilities". That's all you need to know.

3. **Think in terms of data flow** - Each layer enriches the representation. By the final layer, each word "knows about" all other words.

4. **The decoder is just an encoder with blinders** - It can't look ahead, which makes it suitable for generation.

5. **It's all differentiable** - Every operation can be backpropagated through, so standard gradient descent works.

## Practical Next Steps

1. **Visualization** - Add code to plot attention matrices. Seeing is believing.

2. **Small experiments** - Try a tiny model (embed_dim=16, num_heads=2) on simple tasks like reversing sequences.

3. **Profiling** - Most time is spent in matrix multiplications. Understanding this helps with optimization.

4. **Play with hyperparameters** - Change embed_dim, num_heads, num_layers and observe the effects on model capacity and speed.

The beauty of Transformers is they're conceptually simple - just attention layers stacked together. The complexity comes from engineering optimizations (like multi-head attention) that make them work well in practice.

URLs:
- https://arxiv.org/pdf/1706.03762
- https://github.com/ankane/torch.rb
