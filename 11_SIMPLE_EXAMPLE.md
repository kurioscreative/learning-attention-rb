# Understanding the Simple Example

## Putting Theory into Practice

This file demonstrates the README's practical advice: "Try a tiny model (embed_dim=16, num_heads=2) on simple tasks." It shows a complete working example of training and using a Transformer.

## The Three Key Components

### 1. SimpleGenerator: Text Generation in Action

```ruby
class SimpleGenerator
  def generate(start_token, max_length = 50)
    output = Torch.tensor([[start_token]])
    
    Torch.no_grad do
      max_length.times do
        predictions = @model.forward(output, output)
        next_token = predictions[0, -1, 0..-1].argmax.item
        output = Torch.cat([output, Torch.tensor([[next_token]])], dim: 1)
        break if next_token == 2  # End token
      end
    end
    
    output[0].to_a
  end
end
```

#### What's Happening Here?

**Autoregressive Generation** - The decoder generating one token at a time:
1. Start with a seed token
2. Run model to predict next token
3. Append prediction to input
4. Repeat until end token or max length

**Key Engineering Details:**
- `Torch.no_grad`: No gradients needed during inference (saves memory)
- `argmax.item`: Greedy decoding (always pick most likely token)
- `break if next_token == 2`: Stop at end token (hardcoded as 2)

### 2. Training Step: Teacher Forcing

```ruby
def train_step(model, src, tgt, optimizer)
  # Shift target for training
  tgt_input = tgt[0..-1, 0...-1]   # [START, A, B, C]
  tgt_output = tgt[0..-1, 1..-1]   # [A, B, C, END]
  
  predictions = model.forward(src, tgt_input)
  
  loss = Torch::NN::Functional.cross_entropy(
    predictions.reshape(-1, predictions.size(-1)),
    tgt_output.reshape(-1)
  )
```

#### The Teacher Forcing Pattern

This implements the standard Transformer training approach:

**Input/Output Shift:**
```
Given sequence: [START, The, cat, sat, END]
Model input:    [START, The, cat, sat]
Model target:   [The, cat, sat, END]
```

The model learns to predict the next token given all previous tokens.

**Why This Works:**
- During training: Model sees correct previous tokens (teacher forcing)
- During inference: Model uses its own predictions
- This mismatch can cause "exposure bias" but works well in practice

### 3. Usage Example: Seeing It All Together

```ruby
# Create model
model = SimpleTransformer.new(
  vocab_size: 1000,
  embed_dim: 128,
  num_heads: 4
)

# Example data
src = Torch.randint(0, 1000, [2, 10])  # Batch of 2, length 10
tgt = Torch.randint(0, 1000, [2, 8])   # Batch of 2, length 8

# Forward pass
output = model.forward(src, tgt)
```

## Understanding the Data Flow

### Shape Analysis
```ruby
puts "Source shape: #{src.shape}"      # [2, 10] - 2 sequences, 10 tokens
puts "Target shape: #{tgt.shape}"      # [2, 8]  - 2 sequences, 8 tokens
puts "Output shape: #{output.shape}"   # [2, 8, 1000] - logits for each position
```

The output shape `[2, 8, 1000]` means:
- Batch size: 2
- Sequence length: 8
- Vocabulary size: 1000 (probabilities for each token)

### The Database Query Analogy in Action

During generation:
```sql
-- Step 1: Start with [1]
SELECT most_likely_next_token 
FROM vocabulary 
WHERE context = [1]

-- Step 2: Now have [1, 523]
SELECT most_likely_next_token 
FROM vocabulary 
WHERE context = [1, 523]

-- Continue until END token...
```

## What's Simplified (And Why It Matters)

The code explicitly lists what's missing:

### 1. **Cross-Attention**
```ruby
# Current: decoder(tgt) 
# Should be: decoder(tgt, encoder_output)
```
Without this, the decoder can't "see" the input - it's just a language model.

### 2. **Masked Attention**
The decoder should only see previous tokens:
```ruby
# Missing: causal mask to prevent looking ahead
mask = Torch.ones(seq_len, seq_len).triu(1) * -Float::INFINITY
```

### 3. **Better Position Encodings**
Current: Fixed sinusoidal
Better options:
- Learned embeddings (like BERT)
- Relative positions (like T5)
- RoPE (Rotary Position Embeddings)

### 4. **Training Tricks**
Professional implementations include:
- Learning rate warmup and decay
- Label smoothing (prevents overconfidence)
- Gradient clipping (prevents explosions)
- Mixed precision training (faster)

### 5. **Inference Optimizations**
```ruby
# Current: Recompute everything each step
# Better: Cache previous key/value pairs
@kv_cache[layer_idx] = {keys: k, values: v}
```

## Key Engineering Insights Demonstrated

The code summarizes the core insights beautifully:

### 1. **"Attention is just matrix multiplication + softmax"**
No magic - just linear algebra:
```ruby
scores = Q @ K.T
weights = softmax(scores)
output = weights @ V
```

### 2. **"Multi-head = Running attention in parallel"**
Like the README's regex analogy - different heads find different patterns.

### 3. **"Residual connections = 'x + layer(x)'"**
Prevents vanishing gradients in deep networks:
```ruby
# Instead of: x = layer(x)
# Do: x = x + layer(x)
```

### 4. **"Position encoding = Hack to add order"**
Without it, "cat eats fish" = "fish eats cat" to the model.

### 5. **"The whole thing is differentiable"**
Every operation has a gradient - standard backprop works!

## Practical Experiments to Try

Following the README's suggestions:

### 1. **Sequence Reversal Task**
```ruby
# Train to reverse sequences
# Input: [1, 2, 3, 4] ’ Output: [4, 3, 2, 1]
model = SimpleTransformer.new(
  vocab_size: 10,  # Just digits
  embed_dim: 16,   # Tiny
  num_heads: 2     # Minimal
)
```

### 2. **Visualization**
```ruby
# Modify forward to return attention weights
predictions, attention_weights = model.forward(src, tgt, return_attention: true)
# Plot attention_weights as heatmap
```

### 3. **Profiling**
```ruby
require 'benchmark'
Benchmark.bm do |x|
  x.report("forward:") { model.forward(src, tgt) }
  x.report("generate:") { generator.generate(1) }
end
```

## The Learning Journey

This example perfectly demonstrates the README's philosophy:

1. **Start Simple**: Basic model with minimal features
2. **Make It Work**: Can train and generate (even if poorly)
3. **Understand Limitations**: Explicitly lists what's missing
4. **Clear Next Steps**: Add cross-attention, masking, optimizations

## Key Takeaways

1. **Generation is Sequential**: One token at a time, using previous predictions
2. **Training Uses Teacher Forcing**: Model sees correct tokens during training
3. **Many Optimizations Exist**: But the core concept is simple
4. **It's All Just Tensors**: No magic, just matrix operations
5. **Experimentation is Key**: Try the suggested experiments!

As the README emphasizes: "The beauty of Transformers is they're conceptually simple." This example proves it - in ~100 lines, we have a working (if simplified) Transformer that can train and generate text.