# AI/ML Vocabulary for Engineers

This guide explains AI/ML terms through familiar engineering concepts. Think of neural networks as sophisticated data pipelines where information flows through transformations.

## Core Mathematical Operations

### Dot Product
Two vectors multiplied element-wise, then summed. Like calculating correlation between two signals.
```
[1, 2, 3] • [4, 5, 6] = 1×4 + 2×5 + 3×6 = 32
```
**Intuition**: Measures how much two vectors "align" - larger values mean more similarity.

### Matrix Multiplication
Combining rows and columns to transform data dimensions. The fundamental operation for linear transformations.
```
[1 2] × [5 6] = [1×5+2×7  1×6+2×8] = [19 22]
[3 4]   [7 8]   [3×5+4×7  3×6+4×8]   [43 50]
```
**Intuition**: Like applying multiple linear transformations at once - reshaping data through weighted combinations.

### Exponentiation (exp)
e^x operation, used in softmax and other normalization functions.
**Intuition**: Amplifies differences between values exponentially - small differences become large.

## Neural Network Building Blocks

### Tensor
Multi-dimensional array - the fundamental data structure in deep learning.
- 1D tensor: vector [1, 2, 3]
- 2D tensor: matrix [[1, 2], [3, 4]]
- 3D tensor: cube of numbers (e.g., sequence of word embeddings)
**Intuition**: Think of it as a generalized array that can hold data in any number of dimensions.

### Neural Network (Torch::NN)
Ruby's torch.rb equivalent of PyTorch's nn module. Contains pre-built layers and models.
**Intuition**: A toolbox of LEGO blocks for building neural networks.

### Linear Layer (Torch::NN::Linear)
```ruby
Torch::NN::Linear.new(input_size, output_size)
```
Performs matrix multiplication + bias: `output = input × weights + bias`
**Intuition**: Like a programmable filter that learns to transform inputs into useful representations.

### Activation Functions

#### ReLU (Rectified Linear Unit)
```ruby
Torch::NN::ReLU.new
# ReLU([-2, 0, 3.5]) = [0, 0, 3.5]
```
Sets negative values to 0, passes positive values unchanged.
**Intuition**: Acts like a one-way valve - only lets positive signals through. This simple nonlinearity enables networks to learn complex patterns.

#### Softmax
```ruby
Torch::NN::Functional.softmax(logits, dim: -1)
```
Converts raw scores into probabilities that sum to 1.
**Intuition**: Like normalizing vote counts into percentages - turns arbitrary numbers into a probability distribution.

### Layer Normalization (LayerNorm)
```ruby
Torch::NN::LayerNorm.new(embed_dim)
```
Normalizes features to have mean=0, variance=1 across the feature dimension.
**Intuition**: Like auto-adjusting audio levels - keeps signals in a stable range for easier processing.

### Dropout
```ruby
Torch::NN::Dropout.new(p: 0.1)  # Drop 10% of connections
```
Randomly zeros out some connections during training.
**Intuition**: Like randomly removing team members during practice - forces the system to be robust and not rely on any single connection.

### Embedding
```ruby
Torch::NN::Embedding.new(vocab_size, embed_dim)
```
Converts discrete tokens (like words) into continuous vectors.
**Intuition**: Like assigning GPS coordinates to cities - gives each discrete item a position in continuous space where similar items are nearby.

## Torch Operations

### torch.tensor
Creates a tensor from Ruby arrays.
```ruby
Torch.tensor([1, 2, 3])
```

### torch.no_grad
Disables gradient computation for efficiency during inference.
```ruby
Torch.no_grad { model.forward(input) }
```
**Intuition**: Like putting the system in "read-only" mode - faster when you don't need to track changes for learning.

### torch.cat
Concatenates tensors along a dimension.
```ruby
Torch.cat([tensor1, tensor2], dim: 0)
```
**Intuition**: Like joining arrays but works in any dimension.

## Training Operations

### Cross Entropy Loss
```ruby
Torch::NN::Functional.cross_entropy(predictions, targets)
```
Measures how wrong the predictions are compared to true labels.
**Intuition**: Like a scoring system that heavily penalizes confident wrong answers.

### optimizer.zero_grad
```ruby
optimizer.zero_grad
```
Clears gradients from previous step.
**Intuition**: Like erasing the whiteboard before solving a new problem - prevents accumulation of old calculations.

### loss.backward
```ruby
loss.backward
```
Computes gradients - how much each parameter contributed to the error.
**Intuition**: Like tracing back through a supply chain to find which supplier caused a defect.

### optimizer.step
```ruby
optimizer.step
```
Updates model parameters based on computed gradients.
**Intuition**: Like adjusting knobs based on feedback - each parameter moves in the direction that reduces error.

### loss.item
```ruby
loss.item
```
Extracts the scalar value from a single-element tensor.
**Intuition**: Like unwrapping a boxed value to get the actual number.

## Transformer-Specific Concepts

### Attention
The mechanism that lets the model focus on relevant parts of the input.
**Intuition**: Like having dynamic pointers that can look at any part of the data based on what's currently being processed.

### Multi-Head Attention
Running multiple attention mechanisms in parallel with different "perspectives."
**Intuition**: Like having multiple experts examine the same data from different angles, then combining their insights.

### Positional Encoding
Adding position information to embeddings since transformers don't inherently understand sequence order.
**Intuition**: Like adding timestamps to messages so you know their order even if they arrive out of sequence.

## Sequential and ModuleList
```ruby
Torch::NN::Sequential.new(layer1, layer2, layer3)
Torch::NN::ModuleList.new([layer1, layer2, layer3])
```
**Sequential**: Chains layers where output of one feeds into the next.
**ModuleList**: Container for layers that might be used in more complex patterns.
**Intuition**: Sequential is like a Unix pipe (`|`), ModuleList is like an array of functions you can call in any order.