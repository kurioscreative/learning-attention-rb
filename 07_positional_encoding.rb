require 'torch'

# ============================================================================
# LEVEL 7: Positional Encoding
# "Transformers don't know word order, so we add position info"
# ============================================================================
class PositionalEncoding < Torch::NN::Module
  def initialize(embed_dim, max_len = 5000)
    super()

    # Create a matrix to store positional encodings
    pe = Torch.zeros(max_len, embed_dim)
    position = Torch.arange(0, max_len).unsqueeze(1).float

    # Create div_term for the sinusoidal pattern
    # This creates different frequencies for each dimension
    div_term = Torch.exp(
      Torch.arange(0, embed_dim, 2).float *
      -(Math.log(10_000.0) / embed_dim)
    )

    # Apply sin to even indices (0, 2, 4, 6...)
    # Each even dimension gets a different frequency
    (0...embed_dim).step(2).each_with_index do |dim, freq_idx|
      # For this dimension, calculate sin for all positions
      angles = position * div_term[freq_idx] # [max_len, 1] * scalar = [max_len, 1]
      pe[0..-1, dim] = Torch.sin(angles).squeeze(1) # Remove extra dimension
    end

    # Apply cos to odd indices (1, 3, 5, 7...)
    # Same frequencies as even indices, but using cos instead of sin
    (1...embed_dim).step(2).each_with_index do |dim, freq_idx|
      # For this dimension, calculate cos for all positions
      angles = position * div_term[freq_idx] # [max_len, 1] * scalar = [max_len, 1]
      pe[0..-1, dim] = Torch.cos(angles).squeeze(1) # Remove extra dimension
    end

    # Register as buffer (not trainable but moves with model)
    register_buffer('pe', pe.unsqueeze(0))
  end

  def forward(x)
    # Add positional encoding to embeddings
    seq_len = x.size(1)
    x + @pe[0, 0...seq_len, 0..-1]
  end
end

if __FILE__ == $PROGRAM_NAME
  puts '=== Level 7: Positional Encoding ==='

  # Create positional encoding for small embeddings
  pe = PositionalEncoding.new(8, 10) # 8-dim embeddings, max 10 positions

  # Create fake embeddings (batch=1, seq_len=4, embed_dim=8)
  embeddings = Torch.ones(1, 4, 8) * 0.1

  puts 'Original embeddings (all same value 0.1):'
  puts embeddings[0, 0, 0...4].to_a.map { |v| v.round(3) }.join(', ') + '...'

  # Add positional encoding
  output = pe.forward(embeddings)

  puts "\nAfter adding positional encoding:"
  puts "Position 0: #{output[0, 0, 0...4].to_a.map { |v| v.round(3) }.join(', ')}..."
  puts "Position 1: #{output[0, 1, 0...4].to_a.map { |v| v.round(3) }.join(', ')}..."
  puts "Position 2: #{output[0, 2, 0...4].to_a.map { |v| v.round(3) }.join(', ')}..."

  puts "\nKey insights:"
  puts '- Each position gets a unique pattern added'
  puts '- Uses sin/cos at different frequencies'
  puts '- Allows model to distinguish word positions'
  puts "- Without this, 'cat eats fish' = 'fish eats cat'!"
end
