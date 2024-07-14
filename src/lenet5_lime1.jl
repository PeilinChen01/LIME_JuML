using Flux
using MLDatasets
using DataFrames
using Random
using StatsBase
using Plots
using Lasso
using LinearAlgebra  # Importing the LinearAlgebra package for norm function
using ExplainableAI

# Load the FashionMNIST dataset
train_data = MLDatasets.FashionMNIST(split=:train)
test_data = MLDatasets.FashionMNIST(split=:test)

# Extract data and labels
train_X = Float32.(train_data.features) / 255.0
train_y = Flux.onehotbatch(train_data.targets .+ 1, 1:10)
test_X = Float32.(test_data.features) / 255.0
test_y = Flux.onehotbatch(test_data.targets .+ 1, 1:10)

# Reshape the data to (28, 28, 1, N)
train_X = reshape(train_X, 28, 28, 1, :)
test_X = reshape(test_X, 28, 28, 1, :)

# Define the LeNet-5 model
model = Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 16 => 120, relu),  # Adjusted convolution kernel size
    Flux.flatten,
    Dense(120 * 2 * 2, 84, relu),   # Ensuring the input size of the fully connected layer is correct
    Dense(84, 10),
    softmax
)

# Load pre-trained model weights (assuming the weights are saved in 'lenet5_model.bson')
using BSON: @load
@load "lenet5_model1.bson" model

# Select a test instance to explain
n = 5
x = test_X[:, :, :, n]
y_true = test_y[:, n]

# Function to generate perturbed samples
function sample_around_image(x, num_samples=100, noise_level=0.1)
    z = []
    for _ in 1:num_samples
        push!(z, x .+ noise_level * randn(size(x)))
    end
    return stack(z)
end

# Generate perturbed samples
num_samples = 100
Z = sample_around_image(x, num_samples)

# Predict using the trained model
f_Z = model(Z)

# Verify model output
println("Shape of f_Z: ", size(f_Z))

# Extract the predictions for the specific class (e.g., class 1)
f_Z_class = f_Z[1, :]  # Ensure it's a vector of length 100

# Flatten the perturbed samples for linear regression
Z_flat = reshape(Z, num_samples, 28*28)  # Changed to match the expected input format for Lasso

# Ensure the lengths of the columns match the number of samples
println("Number of samples: ", num_samples)
println("Shape of Z_flat: ", size(Z_flat))
println("Length of f_Z_class: ", length(f_Z_class))

# Convert DataFrame to matrix form for Lasso
X = Z_flat
y = convert(Vector{Float64}, f_Z_class)  # Convert y to Float64 to match X

# Print the shapes of X and y
println("Shape of X: ", size(X))
println("Length of y: ", length(y))

# Fit the Lasso model using Lasso.jl
lasso_model = fit(LassoPath, X, y, Normal(), IdentityLink())

display(lasso_model)

lambda_values = lasso_model.λ

# Extract the coefficients
β_opt = coef(lasso_model)

# Verify the length of β_opt
println("Length of β_opt: ", length(β_opt))

# Plot the λ path
p1 = plot(lambda_values, β_opt', xscale=:log10, xlabel="λ (log scale)", ylabel="Coefficients", title="Lasso Path", legend=:bottomleft)
display(p1)

# Ensure K is not larger than the number of features

# K = min(100, length(β_opt[:, end]))
K = min(100, length(β_opt[:, end]))

# Sort the coefficients by their absolute values and select the top K
top_k_indices = sortperm(abs.(β_opt[:, end][2:end]), rev=true)[1:K]  # Ignore the intercept term

# Extract the top K coefficients
top_k_coefs = β_opt[top_k_indices .+ 1, end]  # Offset by one to match the ignored intercept term indices

# Display the heatmap
image = reshape(x, 28, 28)
heatmap_data = zeros(28, 28)

# Map the top K coefficients to their corresponding positions
for idx in top_k_indices
    row = div(idx - 1, 28) + 1
    col = (idx - 1) % 28 + 1
    heatmap_data[row, col] = β_opt[idx + 1, end]  # Fill in the heatmap with the coefficient values
end

# Plot the heatmap without the image as the background
p2 = heatmap(heatmap_data, title="Heatmap of FashionMNIST Image with Top 100 Coefficients")
display(p2)

return top_k_coefs
