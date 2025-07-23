require 'torch'
require_relative './05_multi_head_attention'
# ============================================================================
# LEVEL 6: Transformer Block
# "Attention + Feed-Forward + Residual Connections"
# ============================================================================

class TransformerBlock < Torch::NN::Module
  def initialize(embed_dim, num_heads, ff_dim, dropout = 0.1)
    super()

    # Multi-head attention
    @attention = MultiHeadAttention.new(embed_dim, num_heads)

    # Feed-forward network (just 2 linear layers with ReLU)
    @ff = Torch::NN::Sequential.new(
      Torch::NN::Linear.new(embed_dim, ff_dim),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(ff_dim, embed_dim)
    )

    # Layer normalization (like batch norm but for sequences)
    @norm1 = Torch::NN::LayerNorm.new(embed_dim)
    @norm2 = Torch::NN::LayerNorm.new(embed_dim)

    # Dropout for regularization
    @dropout = Torch::NN::Dropout.new(p: dropout)
  end

  def forward(x)
    # Self-attention with residual connection
    attn_output, = @attention.forward(x, x, x)
    x = @norm1.call(x + @dropout.call(attn_output))

    # Feed-forward with residual connection
    ff_output = @ff.call(x)
    @norm2.call(x + @dropout.call(ff_output))
  end
end

# Test it
if __FILE__ == $PROGRAM_NAME
  puts '=== Level 6: Transformer Block ==='
  block = TransformerBlock.new(8, 2, 32)
  x = Torch.randn(1, 3, 8)
  output = block.forward(x)
  puts "Input shape: #{x.shape}"
  puts "Output shape: #{output.shape}"
  puts 'This is the core building block!'
  puts
end
