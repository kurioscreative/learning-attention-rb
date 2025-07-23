require 'torch'
require_relative './08_transformer_encoder'
require_relative './09_transformer_decoder'

# ============================================================================
# LEVEL 10: Complete Transformer
# "Put it all together"
# ============================================================================
class SimpleTransformer < Torch::NN::Module
  def initialize(vocab_size:, embed_dim: 128, num_heads: 4, ff_dim: 512, num_layers: 2)
    super()

    @encoder = TransformerEncoder.new(
      vocab_size: vocab_size,
      embed_dim: embed_dim,
      num_heads: num_heads,
      ff_dim: ff_dim,
      num_layers: num_layers
    )

    @decoder = TransformerDecoder.new(
      vocab_size: vocab_size,
      embed_dim: embed_dim,
      num_heads: num_heads,
      ff_dim: ff_dim,
      num_layers: num_layers
    )
  end

  def forward(src, tgt)
    # Encode source sequence
    memory = @encoder.forward(src)

    # Decode target sequence (simplified - no cross-attention yet)
    @decoder.forward(tgt)
  end
end

# ============================================================================
# Minimal Test Output
# ============================================================================

if __FILE__ == $0
  puts "=== Simple Transformer Test ==="
  
  # Create a small transformer
  model = SimpleTransformer.new(
    vocab_size: 100,
    embed_dim: 16,
    num_heads: 2,
    ff_dim: 64,
    num_layers: 1
  )
  
  puts "\nModel created with:"
  puts "  vocab_size: 100"
  puts "  embed_dim: 16"
  puts "  num_heads: 2"
  puts "  ff_dim: 64"
  puts "  num_layers: 1"
  
  # Create sample data
  src = Torch.tensor([[5, 23, 67, 89]], dtype: :long)  # "The cat sat on"
  tgt = Torch.tensor([[1, 45, 78]], dtype: :long)      # "Le chat s'est"
  
  puts "\n--- Input Data ---"
  puts "Source: #{src[0].to_a.inspect} (shape: #{src.shape})"
  puts "Target: #{tgt[0].to_a.inspect} (shape: #{tgt.shape})"
  
  # Run encoder
  puts "\n--- Encoder Output ---"
  encoder_output = model.instance_variable_get(:@encoder).forward(src)
  puts "Encoder output shape: #{encoder_output.shape}"
  puts "  (batch_size=1, seq_len=4, embed_dim=16)"
  
  # Show first few values
  puts "First position embedding (truncated):"
  puts "  #{encoder_output[0, 0, 0...5].to_a.map{|v| v.round(3)}.inspect}..."
  
  # Run decoder (currently without cross-attention)
  puts "\n--- Decoder Output ---"
  decoder_output = model.instance_variable_get(:@decoder).forward(tgt)
  puts "Decoder output shape: #{decoder_output.shape}"
  puts "  (batch_size=1, seq_len=3, vocab_size=100)"
  
  # Show probability distribution for first position
  probs = Torch::NN::Functional.softmax(decoder_output[0, 0, 0..-1], dim: 0)
  top3_vals, top3_idx = probs.topk(3)
  puts "\nTop 3 predictions for first position:"
  top3_idx.to_a.zip(top3_vals.to_a).each_with_index do |(idx, prob), i|
    puts "  #{i+1}. Token #{idx}: #{(prob * 100).round(1)}%"
  end
  
  # Run full forward pass
  puts "\n--- Full Forward Pass ---"
  output = model.forward(src, tgt)
  puts "Output shape: #{output.shape}"
  
  puts "\n--- Key Observations ---"
  puts "1. Encoder creates context-aware embeddings for each position"
  puts "2. Decoder generates vocabulary distributions for each position"
  puts "3. Currently missing cross-attention (decoder doesn't use encoder output)"
  puts "4. Without training, predictions are random"
  
  # Parameter count
  total_params = model.parameters.map { |p| p.numel }.sum
  puts "\nTotal parameters: #{total_params}"
  puts "  (Compare to GPT-2: 124M, GPT-3: 175B)"
end
