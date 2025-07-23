require 'torch'

# ============================================================================
# LEVEL 3: Matrix Attention (The Real Deal)
# "Each word looks at all words, including itself"
#
# This code demonstrates how attention works at its most fundamental level - as a matrix operation where each word "looks at" all other words in the sequence.
# =====

class MatrixAttention
  def forward(x)
    # x shape: [seq_len, dim]
    # Think of it as each row is a word, each column is a feature

    # Every word (row) needs to look at every word (row)
    # This is just matrix multiplication
    scores = Torch.matmul(x, x.t) # x * x^T

    # Convert each row to possibility
    attention_weights = Torch.softmax(scores, dim: -1)

    # Apply attention
    output = Torch.matmul(attention_weights, x)
    [output, attention_weights]
  end
end

# Test it
puts '=== Level 3: Matrix Attention ==='
matrix = MatrixAttention.new
# 3 words, each with 2 features
x = Torch.tensor([[1.0, 0.0],   # Word 1: [1, 0]
                  [0.0, 1.0],   # Word 2: [0, 1]
                  [0.5, 0.5]])  # Word 3: [0.5, 0.5]
output, weights = matrix.forward(x)
puts "Input shape: #{x.shape}"
puts "Output shape: #{output.shape}"
puts "Attention weights:\n#{weights}"
puts
