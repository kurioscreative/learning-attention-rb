# Understanding the Simple Transformer

## The Complete Architecture: Encoder + Decoder

This is where all the components come together. As the README states: "The beauty of Transformers is they're conceptually simple - just attention layers stacked together."

## What This Code Shows

The `SimpleTransformer` class demonstrates the full Transformer architecture in its most basic form:

```ruby
class SimpleTransformer < Torch::NN::Module
  def initialize(vocab_size:, embed_dim: 128, num_heads: 4, ff_dim: 512, num_layers: 2)
    super()
    
    @encoder = TransformerEncoder.new(...)
    @decoder = TransformerDecoder.new(...)
  end
end
```

## The Two-Part Architecture

### 1. **Encoder**: Understanding the Input
```ruby
@encoder = TransformerEncoder.new(
  vocab_size: vocab_size,
  embed_dim: embed_dim,
  num_heads: num_heads,
  ff_dim: ff_dim,
  num_layers: num_layers
)
```
- Processes the entire source sequence at once
- Creates rich, context-aware representations
- Bidirectional attention (can see past and future)

### 2. **Decoder**: Generating the Output
```ruby
@decoder = TransformerDecoder.new(
  vocab_size: vocab_size,
  embed_dim: embed_dim,
  num_heads: num_heads,
  ff_dim: ff_dim,
  num_layers: num_layers
)
```
- Generates output one token at a time
- Uses masked attention (can't see future)
- Should attend to encoder output (cross-attention)

## The Forward Pass

```ruby
def forward(src, tgt)
  # Encode source sequence
  memory = @encoder.forward(src)
  
  # Decode target sequence (simplified - no cross-attention yet)
  @decoder.forward(tgt)
end
```

### Current Implementation Status

Looking at this code, it's a **simplified version** that's missing a crucial component:

**Missing: Cross-Attention Connection**
- The decoder receives `tgt` but not `memory` from the encoder
- This means the decoder can't actually "see" the input
- It's essentially two independent models

### What It Should Be

```ruby
def forward(src, tgt)
  # Encode source sequence
  memory = @encoder.forward(src)
  
  # Decode target sequence using encoder output
  @decoder.forward(tgt, memory)  # Pass memory for cross-attention
end
```

## The Data Flow Pipeline

Using the README's mental model:

### Complete Flow:
```ruby
source_tokens
  |> encoder           # Create context-rich representations
  |> memory           # Store encoder output
  
target_tokens + memory
  |> decoder          # Generate output conditioned on input
  |> predictions      # Token probabilities
```

### Concrete Example: Translation

```ruby
# English to French
src = [5, 23, 67]        # "The cat sat"
tgt = [101, 102, 103]    # "Le chat s'est"

# Step 1: Encode source
memory = @encoder.forward(src)
# memory now contains rich representations of "The cat sat"

# Step 2: Decode with memory
output = @decoder.forward(tgt, memory)
# output predicts next tokens: "chat", "s'est", "assis"
```

## Default Hyperparameters Analysis

```ruby
embed_dim: 128    # Moderate size - good for experiments
num_heads: 4      # Each head sees 32 dims (128/4)
ff_dim: 512       # 4x expansion (typical is 4x embed_dim)
num_layers: 2     # Shallow - good for learning/debugging
```

These defaults follow the README's advice: "Try a tiny model (embed_dim=16, num_heads=2) on simple tasks"

## Memory and Computation

For these default settings:
- Embedding matrix: vocab_size × 128
- Per encoder layer: ~400K parameters
- Per decoder layer: ~600K parameters (with cross-attention)
- Total: ~2-3M parameters (depending on vocab_size)

This is tiny compared to real models:
- GPT-2: 124M parameters
- GPT-3: 175B parameters
- But perfect for understanding!

## What's Missing for a Complete Implementation

### 1. **Cross-Attention in Decoder**
```ruby
# Decoder forward should be:
def forward(tgt, encoder_output, tgt_mask=nil)
  # ... existing code ...
  
  # In each decoder layer:
  x = self_attention(x, x, x, tgt_mask)
  x = cross_attention(x, encoder_output, encoder_output)  # Missing!
  x = feedforward(x)
end
```

### 2. **Proper Masking**
```ruby
# Create causal mask for target
tgt_mask = create_causal_mask(tgt.size(1))
```

### 3. **Output Projection**
```ruby
# In decoder:
@output_projection = Torch::NN::Linear.new(embed_dim, vocab_size)
```

## Use Cases for This Architecture

### 1. **Sequence-to-Sequence Tasks**
- Translation: English ’ French
- Summarization: Long text ’ Short summary
- Question Answering: Context + Question ’ Answer

### 2. **The Database Query Analogy Applied**
```sql
-- Encoder: Understand the source
SELECT context_vectors 
FROM source_words 
WHERE all_positions_visible = true

-- Decoder: Generate conditioned on source
SELECT next_word 
FROM vocabulary 
WHERE context_from_encoder = memory
  AND previous_outputs = generated_so_far
  AND future_positions_masked = true
```

## Practical Next Steps

Following the README's suggestions:

### 1. **Add Visualization**
```ruby
def forward(src, tgt, return_attention=false)
  memory, encoder_attention = @encoder.forward(src, return_attention=true)
  output, decoder_attention = @decoder.forward(tgt, memory, return_attention=true)
  
  if return_attention
    return output, {encoder: encoder_attention, decoder: decoder_attention}
  end
  output
end
```

### 2. **Small Experiments**
```ruby
# Sequence reversal task
model = SimpleTransformer.new(
  vocab_size: 10,      # Just digits 0-9
  embed_dim: 16,       # Tiny
  num_heads: 2,        # Minimal
  ff_dim: 64,          # Small
  num_layers: 1        # Single layer
)

# Train to reverse: [1,2,3] ’ [3,2,1]
```

### 3. **Profile Performance**
```ruby
require 'benchmark'

Benchmark.bm do |x|
  x.report("encoder:") { 100.times { @encoder.forward(src) } }
  x.report("decoder:") { 100.times { @decoder.forward(tgt) } }
end
```

## The Engineering Beauty

This simple file demonstrates:

1. **Modularity**: Encoder and decoder are separate, reusable components
2. **Simplicity**: The full model is just two components combined
3. **Flexibility**: Can use encoder-only (BERT), decoder-only (GPT), or both
4. **Clarity**: The code structure mirrors the conceptual architecture

## Key Takeaways

1. **The Transformer = Encoder + Decoder**
2. **Currently simplified** (missing cross-attention connection)
3. **Perfect for learning** with small default parameters
4. **Demonstrates modularity** of the architecture
5. **Ready for experiments** as suggested in README

As the README emphasizes: Start simple, understand the core concepts, then add complexity. This file is exactly that - the simplest complete Transformer structure, ready for experimentation and learning.