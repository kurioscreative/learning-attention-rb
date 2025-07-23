# Understanding the Transformer Encoder

## The Complete Picture: From Tokens to Context-Rich Representations

The `TransformerEncoder` is the culmination of all previous components, implementing the full encoding pipeline described in the README's mental model.

## The Pipeline Architecture

According to the README, a Transformer Block follows this pipeline:
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

The encoder **stacks multiple of these blocks** to progressively refine representations.

## Component Breakdown

### 1. **Token Embedding Layer**
```ruby
@embedding = Torch::NN::Embedding.new(vocab_size, embed_dim)
```
- Converts token IDs (integers) to dense vectors
- Maps discrete vocabulary (e.g., 10,000 words) to continuous space (e.g., 512 dimensions)
- Like a lookup table: word_id ’ vector

### 2. **Positional Encoding**
```ruby
@pos_encoding = PositionalEncoding.new(embed_dim, max_len)
```
- Adds position information to embeddings
- Without this, "cat eats fish" = "fish eats cat"
- Uses sinusoidal patterns (see Level 7)

### 3. **Transformer Block Stack**
```ruby
@layers = Torch::NN::ModuleList.new(
  num_layers.times.map do
    TransformerBlock.new(embed_dim, num_heads, ff_dim)
  end
)
```
- Multiple layers of attention + feedforward
- Each layer refines the representation
- Deeper layers capture more abstract patterns

## The Forward Pass Flow

### Step 1: Token ’ Embedding
```ruby
x = @embedding.call(x)  # [batch, seq_len] ’ [batch, seq_len, embed_dim]
```
Example: [5, 23, 67] ’ [[0.1, -0.3, ...], [0.5, 0.2, ...], [-0.1, 0.4, ...]]

### Step 2: Scale Embeddings
```ruby
x *= Math.sqrt(x.size(-1))
```
- Scales by embed_dim
- Helps with training stability
- Makes embeddings comparable in magnitude to positional encodings

### Step 3: Add Position Information
```ruby
x = @pos_encoding.forward(x)
```
- Each position gets unique sinusoidal pattern added
- Now model knows word order

### Step 4: Apply Dropout
```ruby
x = @dropout.call(x)
```
- Randomly zeros some values during training
- Prevents overfitting

### Step 5: Pass Through Transformer Blocks
```ruby
@layers.each { |layer| x = layer.forward(x) }
```
Each block does:
1. **Multi-head attention**: Each word looks at all other words
2. **Residual + norm**: Preserves original info + stabilizes
3. **Feedforward**: Individual processing of enriched representations
4. **Residual + norm**: Again preserves + stabilizes

## What Each Layer Accomplishes

### Layer 1: Local Patterns
- Captures immediate context
- Learns basic syntactic relationships
- "The" often precedes nouns

### Layer 2+: Progressively Abstract
- Combines patterns from previous layers
- Builds higher-level understanding
- Can capture long-range dependencies

## The Database Query Analogy Evolution

From the README's mental model:

**Layer 1**: Basic queries
```sql
SELECT weighted_avg(values) 
FROM nearby_words 
WHERE similarity > threshold
```

**Layer N**: Complex queries on enriched data
```sql
SELECT weighted_avg(enriched_values) 
FROM all_words_with_context 
WHERE complex_similarity_patterns > threshold
```

## Key Engineering Insights

### 1. **Stacking = Progressive Refinement**
- Each layer adds more context
- Early layers: syntax, local patterns
- Later layers: semantics, global patterns

### 2. **Residual Connections Are Critical**
- Without them, gradients vanish in deep networks
- Allow direct paths for gradient flow
- Enable very deep models (GPT-3 has 96 layers!)

### 3. **Layer Normalization Stabilizes Training**
- Keeps activations in reasonable range
- Prevents explosion/vanishing of values
- Applied after each sub-layer

### 4. **The Embedding Scaling Trick**
```ruby
x *= Math.sqrt(embed_dim)
```
- Without this, embeddings can be too small relative to positional encoding
- Ensures both contribute meaningfully to final representation

## Practical Example

Input: "The cat sat"
- Token IDs: [5, 23, 67]

After encoder:
- Token 5 ("The"): Contains info about being a determiner, first position, followed by "cat"
- Token 23 ("cat"): Knows it's preceded by "The", is the subject, relates to "sat"
- Token 67 ("sat"): Understands it's the verb, has "cat" as subject

## Memory and Computation

For a typical encoder:
- vocab_size: 50,000
- embed_dim: 512
- num_heads: 8
- ff_dim: 2048
- num_layers: 6

Memory dominated by:
1. Embedding matrix: 50,000 × 512 = 25.6M parameters
2. Each transformer block: ~2.4M parameters
3. Total: ~40M parameters

Computation dominated by:
- Matrix multiplications in attention and feedforward
- Scales quadratically with sequence length (attention)

## Connection to Full Transformer

This encoder is half of the full Transformer:
1. **Encoder**: Processes input, creates representations
2. **Decoder**: Uses encoder output + generates text

For tasks like BERT (understanding), only encoder is needed.
For tasks like GPT (generation), a modified version is used.

## The Beauty of the Design

As the README states: "The beauty of Transformers is they're conceptually simple - just attention layers stacked together."

The encoder demonstrates this:
- Simple components (attention, feedforward, normalization)
- Powerful when stacked
- Each piece has clear purpose
- Together they create state-of-the-art NLP models