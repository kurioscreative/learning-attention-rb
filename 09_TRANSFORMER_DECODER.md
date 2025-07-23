# Understanding the Transformer Decoder

## The Key Insight: "Encoder with Blinders"

From the README: "The decoder is just an encoder with blinders - It can't look ahead, which makes it suitable for generation."

This simple insight captures the essence of the decoder. Let's build on this engineering perspective.

## The Decoder's Job: Autoregressive Generation

While the encoder processes an entire sequence at once (bidirectional), the decoder generates one token at a time, only looking at what came before.

```ruby
# Encoder sees: "The cat sat on the mat"  (all at once)
# Decoder generates: "The" ’ "cat" ’ "sat" ’ "on" ’ "the" ’ "mat"  (step by step)
```

## Component Architecture

The decoder has THREE types of attention layers:

### 1. **Masked Self-Attention** (The "Blinders")
```ruby
@self_attention = MultiHeadAttention.new(embed_dim, num_heads)
```
- Like encoder's self-attention BUT with a mask
- Can only attend to previous positions
- Prevents "cheating" during training

### 2. **Cross-Attention** (The Bridge)
```ruby
@cross_attention = MultiHeadAttention.new(embed_dim, num_heads)
```
- Attends to encoder output
- This is how decoder "reads" the input
- Query from decoder, Keys/Values from encoder

### 3. **Feedforward Network** (Same as Encoder)
```ruby
@feed_forward = Torch::NN::Sequential.new(
  Torch::NN::Linear.new(embed_dim, ff_dim),
  Torch::NN::ReLU.new,
  Torch::NN::Linear.new(ff_dim, embed_dim)
)
```

## The Decoder Block Pipeline

Building on the README's pipeline model:

```ruby
decoder_input
  |> masked_self_attention    # Look at previous outputs only
  |> add_residual            # Preserve original info
  |> normalize               # Stabilize
  |> cross_attention         # Look at encoder output
  |> add_residual           # Preserve
  |> normalize              # Stabilize
  |> feedforward           # Process individually
  |> add_residual         # Preserve
  |> normalize            # Stabilize
  |> output
```

## The Masking Mechanism: Engineering the "Blinders"

### The Problem
During training, we have the full target sequence. Without masking, the model could "cheat" by looking ahead.

### The Solution: Causal Mask
```ruby
def create_causal_mask(seq_len)
  # Create upper triangular matrix of -inf
  mask = Torch.ones(seq_len, seq_len).triu(1) * -Float::INFINITY
  mask
end
```

Visual representation:
```
Position:   1    2    3    4
    1      [0   -   -   -]   # Position 1 can only see position 1
    2      [0    0   -   -]   # Position 2 can see positions 1-2
    3      [0    0    0   -]   # Position 3 can see positions 1-3
    4      [0    0    0    0]   # Position 4 can see all positions
```

## The Generation Process

### Training Mode (Teacher Forcing)
```ruby
# We have: "The cat sat"
# Target: "cat sat <end>"

# Step 1: Input "The" ’ Predict "cat"
# Step 2: Input "The cat" ’ Predict "sat"
# Step 3: Input "The cat sat" ’ Predict "<end>"
```

### Inference Mode (Actual Generation)
```ruby
# Start with <start> token
output = ["<start>"]

while output.last != "<end>" && output.length < max_length
  # Run decoder on all previous outputs
  logits = decoder.forward(output, encoder_output)
  
  # Get prediction for next token
  next_token = logits[-1].argmax  # Or sample
  output.append(next_token)
end
```

## Cross-Attention: The Database Query Evolution

Using the README's database analogy:

### Self-Attention (Masked)
```sql
SELECT weighted_avg(values) 
FROM previous_outputs 
WHERE position <= current_position
  AND similarity(query, key) > threshold
```

### Cross-Attention
```sql
SELECT weighted_avg(encoder_values) 
FROM encoder_outputs 
WHERE similarity(decoder_query, encoder_key) > threshold
```

This is how the decoder "reads" the input while generating output.

## Key Engineering Insights

### 1. **Why Separate Encoder and Decoder?**
- **Encoder**: Bidirectional context for understanding
- **Decoder**: Unidirectional generation
- Separation allows each to optimize for its task

### 2. **The Residual Highway**
Each decoder block has THREE residual connections:
1. Around masked self-attention
2. Around cross-attention
3. Around feedforward

This creates multiple gradient highways for stable training.

### 3. **Layer Coordination**
- Early decoder layers: Focus on language modeling (what word comes next)
- Middle layers: Balance between following patterns and attending to input
- Late layers: Fine-tune based on encoder information

### 4. **The Start Token Trick**
```ruby
# Always begin generation with special <start> token
# This gives the first real token something to attend to
```

## Memory and Computation Complexity

### Decoder-Specific Costs:
1. **Masked Attention**: O(n²) but only lower triangle
2. **Cross-Attention**: O(n_decoder × n_encoder)
3. **Caching**: Can cache previous computations during generation

### Generation Optimization:
```ruby
# Cache key-value pairs from previous steps
@kv_cache = {}

def forward_with_cache(x, step)
  if step > 0
    # Reuse previous computations
    k, v = @kv_cache[step-1]
    # Only compute for new token
  end
end
```

## Practical Example: Translation

Input (Encoder): "The cat sat"
Target: "Le chat s'est assis"

### Training Time:
```
Decoder Input:    ["<start>", "Le", "chat", "s'est"]
Decoder Target:   ["Le", "chat", "s'est", "assis"]
Encoder Output:   [context vectors for "The cat sat"]
```

### Generation Time:
```
Step 1: <start> + encoder_output ’ "Le"
Step 2: <start> Le + encoder_output ’ "chat"
Step 3: <start> Le chat + encoder_output ’ "s'est"
Step 4: <start> Le chat s'est + encoder_output ’ "assis"
```

## The Complete Transformer Decoder Class

Conceptually:
```ruby
class TransformerDecoder
  def initialize(vocab_size, embed_dim, num_heads, ff_dim, num_layers)
    @embedding = Torch::NN::Embedding.new(vocab_size, embed_dim)
    @pos_encoding = PositionalEncoding.new(embed_dim)
    
    @layers = num_layers.times.map do
      DecoderBlock.new(embed_dim, num_heads, ff_dim)
    end
    
    @output_projection = Torch::NN::Linear.new(embed_dim, vocab_size)
  end
  
  def forward(tgt, encoder_output, tgt_mask=nil)
    # Embed and add positions
    x = @embedding.call(tgt) * Math.sqrt(@embed_dim)
    x = @pos_encoding.forward(x)
    
    # Apply decoder blocks
    @layers.each do |layer|
      x = layer.forward(x, encoder_output, tgt_mask)
    end
    
    # Project to vocabulary
    @output_projection.call(x)
  end
end
```

## Connection to Modern Models

### GPT-Style (Decoder-Only)
- Remove cross-attention
- Just masked self-attention + feedforward
- Used for language modeling

### BERT-Style (Encoder-Only)
- Remove masking
- Bidirectional attention
- Used for understanding tasks

### T5/BART (Full Encoder-Decoder)
- Complete architecture as described
- Used for translation, summarization, etc.

## The Beauty of the Design

The decoder demonstrates the power of constraints:
- The masking constraint enables generation
- The cross-attention enables conditioned generation
- The autoregressive nature enables variable-length output

As the README states: "It's all differentiable" - every operation can be trained end-to-end, making this architecture both powerful and elegant.

## Key Takeaways

1. **Decoder = Encoder + Masking + Cross-Attention**
2. **Masking enforces causality** (can't see future)
3. **Cross-attention bridges input and output**
4. **Autoregressive generation** happens one token at a time
5. **Caching makes generation efficient**

The decoder completes the Transformer architecture, enabling tasks that require understanding input (encoder) and generating appropriate output (decoder) - the foundation of modern NLP.