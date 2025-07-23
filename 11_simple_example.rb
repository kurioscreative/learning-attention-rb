require 'torch'
require_relative './10_simple_transformer'

class SimpleGenerator
  def initialize(model, vocab_size)
    @model = model
    @vocab_size = vocab_size
  end

  def generate(start_token, max_length = 50)
    @model.eval

    # Start with a single token
    output = Torch.tensor([[start_token]])

    Torch.no_grad do
      max_length.times do
        # Get predictions
        predictions = @model.forward(output, output)

        # Get next token (greedy decoding)
        next_token = predictions[0, -1, 0..-1].argmax.item

        # Append to output
        output = Torch.cat([
                             output,
                             Torch.tensor([[next_token]])
                           ], dim: 1)

        # Stop if we hit end token (let's say it's 2)
        break if next_token == 2
      end
    end

    output[0].to_a
  end
end

# ============================================================================
# Training Example (Simplified)
# ============================================================================

def train_step(model, src, tgt, optimizer)
  model.train

  # Shift target for training
  # Input: [START, A, B, C]
  # Target: [A, B, C, END]
  tgt_input = tgt[0..-1, 0...-1]
  tgt_output = tgt[0..-1, 1..-1]

  # Forward pass
  predictions = model.forward(src, tgt_input)

  # Calculate loss
  loss = Torch::NN::Functional.cross_entropy(
    predictions.reshape(-1, predictions.size(-1)),
    tgt_output.reshape(-1)
  )

  # Backward pass
  optimizer.zero_grad
  loss.backward
  optimizer.step

  loss.item
end

# ============================================================================
# Usage Example
# ============================================================================

puts '=== Complete Transformer Example ==='

# Create model
model = SimpleTransformer.new(vocab_size: 1000, embed_dim: 128, num_heads: 4)

# Example data
src = Torch.randint(0, 1000, [2, 10], dtype: :long)  # 2 sequences, 10 tokens each
tgt = Torch.randint(0, 1000, [2, 8], dtype: :long)   # 2 sequences, 8 tokens each

# Forward pass
output = model.forward(src, tgt)
puts "Source shape: #{src.shape}"
puts "Target shape: #{tgt.shape}"
puts "Output shape: #{output.shape}"

# Generate some text
generator = SimpleGenerator.new(model, 1000)
generated = generator.generate(1) # Start with token 1
puts "Generated tokens: #{generated.first(10)}..."

# ============================================================================
# NEXT STEPS: What's Missing?
# ============================================================================

puts "\n=== What We Simplified ==="
puts '1. Cross-attention: Decoder attending to encoder output'
puts "2. Masked attention: Preventing decoder from 'cheating'"
puts '3. Better position encodings: Learned or relative positions'
puts '4. Training tricks: Learning rate scheduling, label smoothing'
puts '5. Inference optimizations: Caching, beam search'

puts "\n=== Key Engineering Insights ==="
puts '- Attention is just matrix multiplication + softmax'
puts '- Multi-head = Running attention in parallel with different weights'
puts "- Residual connections = 'x + layer(x)' prevents vanishing gradients"
puts '- Position encoding = Hack to add sequence order information'
puts '- The whole thing is differentiable = Can train with backprop'

# ============================================================================
# Minimal Test Output
# ============================================================================

puts "\n=== Minimal Test Output ==="

# Show actual tensor values for clarity
puts "\n--- Example Input/Output Tensors ---"
small_src = Torch.tensor([[5, 23, 67, 89]], dtype: :long)
small_tgt = Torch.tensor([[1, 45, 78]], dtype: :long)

puts "Source tokens: #{small_src[0].to_a.inspect}"
puts "Target tokens: #{small_tgt[0].to_a.inspect}"

# Run forward pass
small_output = model.forward(small_src, small_tgt)
puts "Output shape: #{small_output.shape} (batch=1, seq_len=3, vocab_size=1000)"

# Show prediction probabilities for first position
probs = Torch::NN::Functional.softmax(small_output[0, 0, 0..-1], dim: 0)
top5_values, top5_indices = probs.topk(5)
puts "\nTop 5 predictions for position 0:"
top5_indices.to_a.zip(top5_values.to_a).each_with_index do |(idx, prob), i|
  puts "  #{i+1}. Token #{idx}: #{(prob * 100).round(2)}%"
end

# Demonstrate training
puts "\n--- Training Step Demo ---"
optimizer = Torch::Optim::Adam.new(model.parameters, lr: 0.001)

# Initial loss
initial_loss = train_step(model, small_src, small_tgt, optimizer)
puts "Initial loss: #{initial_loss.round(4)}"

# Train for a few steps
5.times do |i|
  loss = train_step(model, small_src, small_tgt, optimizer)
  puts "Step #{i+1} loss: #{loss.round(4)}"
end

# Generation example with detailed output
puts "\n--- Generation Process ---"
generator = SimpleGenerator.new(model, 1000)
puts "Starting generation with token 1..."

# Manually show first few steps
output = Torch.tensor([[1]])
3.times do |i|
  predictions = model.forward(output, output)
  next_token = predictions[0, -1, 0..-1].argmax.item
  output = Torch.cat([output, Torch.tensor([[next_token]])], dim: 1)
  puts "Step #{i+1}: #{output[0].to_a.inspect} (predicted: #{next_token})"
end

puts "\n=== Test Complete ===\n"
puts "This demonstrates:"
puts "1. Forward pass produces vocabulary-sized output for each position"
puts "2. Training reduces loss (model is learning)"
puts "3. Generation works autoregressively (one token at a time)"
puts "4. Without proper training data, outputs are random"
