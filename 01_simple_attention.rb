require 'torch'

# ============================================================================
# LEVEL 1: The Simplest Possible "Attention"
# "Look at all words and decide which ones matter"
# ============================================================================

class SimpleAttention

  def forward(words)
    # Imagine each word is just a number for now
    # words = [0.1, 0.5, 0.3, 0.9]

    # Step 1: How important is each word? (Just use the values themselves)
    importance = Torch.softmax(words, dim:0)

    # Step 2: Weighted average based on importance
    output = (words * importance).sum

    output
  end
end

# Test it
puts "=== Level 1: Simple Attention ==="
simple = SimpleAttention.new
words = Torch.tensor([0.1, 0.5, 0.3, 0.9])
puts "Input: #{words}"
puts "Output: #{simple.forward(words)}"
puts
