# Multi-Head Attention: Looking from Multiple Angles

Multi-head attention is a key innovation that allows the model to attend to information from different representation subspaces simultaneously. As the README explains, it's like running multiple regex patterns on the same text - each head looks for different patterns.

## The Core Concept

Instead of having one attention mechanism, multi-head attention runs several attention operations in parallel:
- Each "head" can specialize in different types of relationships
- Results from all heads are concatenated and mixed together
- This allows the model to capture multiple types of patterns simultaneously

## How It Works

### 1. Linear Projections (lines 15-18)
```ruby
@q_linear = Torch::NN::Linear.new(embed_dim, embed_dim)
@k_linear = Torch::NN::Linear.new(embed_dim, embed_dim)
@v_linear = Torch::NN::Linear.new(embed_dim, embed_dim)
```
First, the input is projected into Query, Key, and Value representations using learned linear transformations.

### 2. Split into Multiple Heads (lines 35-37)
```ruby
q = q.view(batch_size, seq_len, @num_heads, @head_dim).transpose(1, 2)
```
The embeddings are reshaped to create multiple "heads". If `embed_dim=8` and `num_heads=2`, each head works with 4 dimensions.

### 3. Parallel Attention (line 40)
```ruby
attn_output, attn_weights = @attention.forward(q, k, v)
```
Each head performs its own attention computation independently, potentially focusing on different aspects:
- **Head 1** might learn syntactic relationships (subject-verb)
- **Head 2** might capture semantic similarity (cat-dog)
- **Head 3** might focus on positional patterns (first-last word)

### 4. Concatenate and Mix (lines 45-50)
```ruby
attn_output = attn_output.transpose(1, 2).contiguous.view(batch_size, seq_len, @embed_dim)
output = @out_linear.call(attn_output)
```
The outputs from all heads are concatenated back together and passed through a final linear layer to mix the information.

## Example in the Code

```ruby
mha = MultiHeadAttention.new(8, 2)  # 8 dims, 2 heads
```
This creates a multi-head attention module where:
- Total embedding dimension: 8
- Number of heads: 2
- Each head works with: 8/2 = 4 dimensions

## Why This Matters

Multi-head attention provides:
1. **Expressiveness**: Different heads can capture different types of relationships
2. **Efficiency**: Parallel computation across heads
3. **Robustness**: If one head fails to capture something important, others might catch it

This is a crucial component that makes Transformers so powerful - they can simultaneously model many different types of relationships in the data.