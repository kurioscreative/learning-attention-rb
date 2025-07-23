require 'torch'
require_relative './06_transformer_block'
require_relative './07_positional_encoding'
# ============================================================================
# LEVEL 9: Simple Decoder
# "Like encoder but with masked attention"
# ============================================================================

class TransformerDecoder < Torch::NN::Module
  def initialize(vocab_size:, embed_dim:, num_heads:, ff_dim:, num_layers:, max_len: 5000)
    super()

    # Same structure as encoder
    @embedding = Torch::NN::Embedding.new(vocab_size, embed_dim)
    @pos_encoding = PositionalEncoding.new(embed_dim, max_len)

    # Decoder blocks (we'll add masking in forward pass)
    @layers = Torch::NN::ModuleList.new(
      num_layers.times.map do
        TransformerBlock.new(embed_dim, num_heads, ff_dim)
      end
    )

    # Output projection
    @output_projection = Torch::NN::Linear.new(embed_dim, vocab_size)
    @dropout = Torch::NN::Dropout.new(p: 0.1)
  end

  def forward(x)
    # Embed and add positions
    x = @embedding.call(x)
    x *= Math.sqrt(x.size(-1))
    x = @pos_encoding.forward(x)
    x = @dropout.call(x)

    # Pass through decoder blocks
    @layers.each { |layer| x = layer.forward(x) }

    # Project to vocabulary
    @output_projection.call(x)
  end
end

# Test the decoder
if __FILE__ == $0
  puts '=== Testing TransformerDecoder ==='

  # Setup
  vocab_size = 1000
  embed_dim = 64
  num_heads = 4
  ff_dim = 256
  num_layers = 2
  batch_size = 2
  seq_len = 10

  # Create decoder
  decoder = TransformerDecoder.new(
    vocab_size: vocab_size,
    embed_dim: embed_dim,
    num_heads: num_heads,
    ff_dim: ff_dim,
    num_layers: num_layers
  )

  # Create sample input (batch of token indices)
  input_tokens = Torch.randint(0, vocab_size, [batch_size, seq_len], dtype: :long)
  puts "Input shape: #{input_tokens.shape}"
  puts "Sample input tokens: #{input_tokens[0].to_a}"

  # Forward pass
  output = decoder.forward(input_tokens)
  puts "\nOutput shape: #{output.shape}"
  puts "Expected: [#{batch_size}, #{seq_len}, #{vocab_size}]"

  # Check output probabilities
  probs = Torch.softmax(output[0, 0], dim: 0)
  puts "\nFirst token probabilities (top 5):"
  values, indices = Torch.topk(probs, 5)
  values.to_a.zip(indices.to_a).each do |prob, idx|
    puts "  Token #{idx}: #{prob.round(4)}"
  end
end
