require 'torch'

class PairwiseAttention
  def forward(query, keys)
    # query: "What am I looking for?" (1 word)
    # keys: "What can I look at?" (multiple words)
    # How similar is the query to each key?
    scores = query * keys # Element-wise multiply

    # Convert to probabilities
    attention_weights = Torch.softmax(scores, dim: 0)

    # Return which words we're paying attention to
    attention_weights
  end
end

# Test it
puts '=== Level 2: Pairwise Attention ==='
pairwise = PairwiseAttention.new
query = Torch.tensor(0.5) # Looking for something like 0.5
keys = Torch.tensor([0.1, 0.5, 0.3, 0.9]) # Available words
attention = pairwise.forward(query, keys)
puts "Query: #{query}"
puts "Keys: #{keys}"
puts "Attention weights: #{attention}"
puts
