require 'torch'

require_relative './06_transformer_block'
require_relative './07_positional_encoding'
# ============================================================================
# LEVEL 8: Complete Encoder
# "Stack multiple transformer blocks"
# ============================================================================

class TransformerEncoder < Torch::NN::Module
  def initialize(vocab_size:, embed_dim:, num_heads:, ff_dim:, num_layers:, max_len: 5000)
    super()

    # Token embedding
    @embedding = Torch::NN::Embedding.new(vocab_size, embed_dim)

    # Positional encoding
    @pos_encoding = PositionalEncoding.new(embed_dim, max_len)

    # Stack of transformer blocks
    @layers = Torch::NN::ModuleList.new(
      num_layers.times.map do
        TransformerBlock.new(embed_dim, num_heads, ff_dim)
      end
    )

    @dropout = Torch::NN::Dropout.new(p: 0.1)
  end

  def forward(x)
    # Convert tokens to embeddings
    x = @embedding.call(x)
    x *= Math.sqrt(x.size(-1)) # Scale embeddings

    # Add positional encoding
    x = @pos_encoding.forward(x)
    x = @dropout.call(x)

    # Pass through transformer blocks
    @layers.each { |layer| x = layer.forward(x) }

    x
  end
end

# Test it
if __FILE__ == $PROGRAM_NAME
  puts '=== Level 8: Transformer Encoder ==='

  # Create a small encoder
  encoder = TransformerEncoder.new(
    vocab_size: 100,
    embed_dim: 8,
    num_heads: 2,
    ff_dim: 32,
    num_layers: 2
  )

  # Create sample input (batch_size=1, seq_len=3)
  # Token IDs: [5, 23, 67] - could represent words
  input_ids = Torch.tensor([[5, 23, 67]])

  puts "Input token IDs: #{input_ids}"
  puts "Input shape: #{input_ids.shape}"

  # Forward pass
  output = encoder.forward(input_ids)

  puts "\nOutput shape: #{output.shape}"
  puts 'Output sample (first token embedding):'
  puts output[0, 0, 0...4].to_a.map { |v| v.round(3) }.join(', ') + '...'

  puts "\nWhat happened:"
  puts '1. Tokens → Embeddings (vocab_size → embed_dim)'
  puts '2. Added positional encoding (so model knows word order)'
  puts '3. Passed through 2 transformer blocks'
  puts '4. Each token now contains information from all other tokens'
end
