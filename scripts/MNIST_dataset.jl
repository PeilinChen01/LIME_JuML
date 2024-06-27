using MLDatasets
using ImageCore, ImageIO, ImageShow

index = 10
x, y = MNIST(Float32, :test)[10]

# By convention in Flux.jl, the input needs to be resized to WHCN format
# by adding a color channel and batch dimensions.
input = reshape(x, 28, 28, 1, :);

convert2image(MNIST, x)