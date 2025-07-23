require 'torch'
require_relative './04_scaled_dot_product_attention'

# ============================================================================
# LEVEL 5: Multi-Head Attention
# "Let's look at the same thing from multiple angles"
# ============================================================================

class MultiHeadAttention < Torch::NN::Module
  def initialize(embed_dim, num_heads)
    super()
    @embed_dim = embed_dim
    @num_heads = num_heads
    @head_dim = embed_dim / num_heads

    # Instead of one attention mechanism, we have multiple "heads"
    # Each head can learn different relationships
    @q_linear = Torch::NN::Linear.new(embed_dim, embed_dim)
    @k_linear = Torch::NN::Linear.new(embed_dim, embed_dim)
    @v_linear = Torch::NN::Linear.new(embed_dim, embed_dim)
    @out_linear = Torch::NN::Linear.new(embed_dim, embed_dim)

    @attention = ScaledDotProductAttention.new
  end

  def forward(query, key, value)
    batch_size = query.size(0)
    seq_len = query.size(1)

    # Transform Q, K, V
    q = @q_linear.call(query)
    k = @k_linear.call(key)
    v = @v_linear.call(value)

    # Reshape for multiple heads
    # From: [batch, seq_len, embed_dim]
    # To: [batch, num_heads, seq_len, head_dim]
    q = q.view(batch_size, seq_len, @num_heads, @head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, @num_heads, @head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, @num_heads, @head_dim).transpose(1, 2)

    # Apply attention on each head
    attn_output, attn_weights = @attention.forward(q, k, v)

    # Concatenate heads
    # From: [batch, num_heads, seq_len, head_dim]
    # To: [batch, seq_len, embed_dim]
    attn_output = attn_output.transpose(1, 2).contiguous.view(
      batch_size, seq_len, @embed_dim
    )

    # Final linear transformation
    output = @out_linear.call(attn_output)

    [output, attn_weights]
  end
end

# Test it
if __FILE__ == $0
  puts '=== Level 5: Multi-Head Attention ==='
  mha = MultiHeadAttention.new(8, 2) # 8 dims, 2 heads
  x = Torch.randn(1, 3, 8) # 1 batch, 3 words, 8 dims
  output, = mha.forward(x, x, x) # Self-attention
  puts "Input shape: #{x.shape}"
  puts "Output shape: #{output.shape}"
  puts 'Each head learns different patterns!'
  puts
end
