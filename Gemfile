# frozen_string_literal: true

source "https://rubygems.org"

git_source(:github) {|repo_name| "https://github.com/#{repo_name}" }

# gem "rails"

# Torch (v0.20) Instructions:
# First, download LibTorch. For Mac arm64, use:
#
# curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip > libtorch.zip
# unzip -q libtorch.zip
# For Linux x86-64, use the build that matches your CUDA version. For other platforms, build LibTorch from source.
#
# Then run:
#
# bundle config build.torch-rb --with-torch-dir=/path/to/libtorch
# And add this line to your application’s Gemfile:
#
# Troubleshooting:
# Libomp may need to be installed as well if you see errors when trying to use torch-rb.
#
gem 'torch-rb'
# It can take 5-10 minutes to compile the extension. Windows is not currently supported.
