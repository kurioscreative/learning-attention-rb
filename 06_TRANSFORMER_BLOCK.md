# Understanding the Transformer Block (Level 6)

## Overview
The Transformer Block is the fundamental building block of the Transformer architecture. Think of it as a pipeline stage that enriches word representations with contextual information while preserving important details through residual connections.

## Key Concept: Pipeline Processing
As described in the README, a Transformer Block follows this pipeline:
```
input
  |> attention_layer      # Enrich with context from all other words
  |> add_residual        # Preserve original information
  |> normalize           # Stabilize numeric values
  |> feedforward        # Process each position individually
  |> add_residual       # Preserve enriched information
  |> normalize          # Stabilize again
  |> output
```

## Component Breakdown

### 1. Multi-Head Attention (`@attention`)
- **What it does**: Allows each word to "look at" all other words and gather relevant information
- **Database analogy**: Like running `SELECT weighted_avg(values) FROM words WHERE similarity(query, key) > threshold`
- **Implementation**: Uses the MultiHeadAttention class from level 5
- **Key insight**: Self-attention means the input queries itself (x queries x using x as keys/values)

### 2. Feed-Forward Network (`@ff`)
- **What it does**: Processes each word position independently after attention
- **Architecture**: Simple 2-layer MLP with ReLU activation
  - Linear layer: embed_dim ’ ff_dim (expansion)
  - ReLU activation (non-linearity)
  - Linear layer: ff_dim ’ embed_dim (compression)
- **Purpose**: Adds computational capacity to process the enriched representations

### 3. Layer Normalization (`@norm1`, `@norm2`)
- **What it does**: Normalizes values across the embedding dimension
- **Why it matters**: Prevents values from exploding or vanishing during deep stacking
- **Different from BatchNorm**: Normalizes across features for each sequence position independently

### 4. Residual Connections
- **Pattern**: `x + dropout(sublayer_output)`
- **Purpose**: Allows information to flow directly through the network
- **Benefits**:
  - Prevents vanishing gradients in deep networks
  - Preserves original information while adding new information
  - Makes optimization easier

### 5. Dropout (`@dropout`)
- **Purpose**: Regularization to prevent overfitting
- **Applied to**: Both attention output and feed-forward output
- **Default rate**: 0.1 (10% dropout)

## The Forward Pass Explained

```ruby
def forward(x)
  # Step 1: Self-attention with residual
  attn_output, = @attention.forward(x, x, x)  # x attends to itself
  x = @norm1.call(x + @dropout.call(attn_output))
  
  # Step 2: Feed-forward with residual
  ff_output = @ff.call(x)
  @norm2.call(x + @dropout.call(ff_output))
end
```

### Step-by-Step:
1. **Self-Attention**: Each position queries all other positions to gather context
2. **Residual + Dropout**: Add the attention output to the original input (with dropout for regularization)
3. **Layer Norm 1**: Normalize the combined representation
4. **Feed-Forward**: Apply position-wise transformation (same for each position)
5. **Residual + Dropout**: Add feed-forward output to the normalized attention output
6. **Layer Norm 2**: Final normalization for stable outputs

## Why This Design Works

### 1. **Information Flow**
- Residual connections ensure information can flow unchanged if needed
- Each sublayer can learn to add information without destroying existing information

### 2. **Complementary Processing**
- Attention: Gathers information across positions (global context)
- Feed-forward: Processes each position individually (local computation)
- Together: Both global and local processing

### 3. **Stability**
- Layer normalization after each sublayer keeps values in a reasonable range
- Crucial for stacking many blocks (typical Transformers have 6-24+ blocks)

### 4. **Modularity**
- Each block is self-contained
- Can stack many blocks to increase model capacity
- Same block structure used in both encoder and decoder

## Test Example Analysis
```ruby
block = TransformerBlock.new(8, 2, 32)
x = Torch.randn(1, 3, 8)
output = block.forward(x)
```

- **embed_dim = 8**: Each word represented by 8 numbers
- **num_heads = 2**: Attention splits into 2 parallel heads
- **ff_dim = 32**: Feed-forward expands to 32 dimensions internally
- **Input shape [1, 3, 8]**: Batch of 1, sequence of 3 words, 8-dimensional embeddings
- **Output shape [1, 3, 8]**: Same shape (residual connections preserve dimensions)

## Engineering Insights

1. **Parameter Efficiency**: Most parameters are in the feed-forward network (2 × embed_dim × ff_dim)
2. **Computational Cost**: Attention is O(n²) in sequence length, feed-forward is O(n)
3. **Memory Usage**: Residual connections require storing intermediate activations
4. **Typical ff_dim**: Usually 4 × embed_dim (e.g., 2048 for embed_dim=512)

## Connection to Full Transformer
- **Encoder**: Stack of these blocks processing the input
- **Decoder**: Similar blocks but with masked self-attention (can't look ahead)
- **Depth**: Original paper used 6 blocks, modern models use 12-96+
- **Each block**: Progressively refines the representation with more context

This is why it's called "the core building block" - everything else in the Transformer is built by stacking and connecting these blocks!