# ============================================================================
# LEVEL 4: Scaled Dot-Product Attention
# "The building block of Transformers"
# ============================================================================

class ScaledDotProductAttention
  def forward(query, key, value)
    # The famous Q, K, V!
    # - Query: What am I looking for?
    # - Key: What can I match against?
    # - Value: What information do I actually want?

    d_k = query.size(-1) # Dimension of the keys

    # Compute attention scores
    scores = Torch.matmul(query, key.transpose(-2, -1))

    # Scale down (this prevents numbers from getting too big)
    scores /= Math.sqrt(d_k)

    # Convert to probabilities
    attention_weights = Torch.softmax(scores, dim: -1)

    # Apply attention to values
    output = Torch.matmul(attention_weights, value)
    [output, attention_weights]
  end
end

if __FILE__ == $PROGRAM_NAME
  # Test it
  puts '=== Level 4: Scaled Dot-Product Attention ==='
  sdpa = ScaledDotProductAttention.new
  # 2 sequences, 3 words each, 4 dimensions
  q = Torch.randn(2, 3, 4)
  k = Torch.randn(2, 3, 4)
  v = Torch.randn(2, 3, 4)
  output, weights = sdpa.forward(q, k, v)
  puts "Q,K,V shape: #{q.shape}"
  puts "Output shape: #{output.shape}"
  puts "Attention shape: #{weights.shape}"
end
