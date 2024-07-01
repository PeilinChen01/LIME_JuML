# lime implementation
using XAIBase
using Random
using LinearAlgebra
using GLMNet # Import GLMNet package for LassoPath
"""
    LIME(model)

Create a LIME analyzer.

TODO: write better documentation.
"""
struct LIME{M} <: AbstractXAIMethod
    model::M
end

# Function: creating N sample points around x_dash
"""
    sample_around(x_dash, N=100, std_dev=0.1)

Create N sample points around x_dash.

# Arguments
x_dash: The center point for sampling.
N: The number of samples to create (default is 100).
std_dev: The standard deviation for the normal distribution used for sampling (default is 0.1).

# Returns
Array of sample points around x_dash.
"""
function sample_around(x_dash, N=100, std_dev=0.1)
    z_dash = [x_dash .+ randn(length(x_dash)) .* std_dev for _ in 1:N]
    return z_dash
end

# Function to calculate the distance between the original x and x_dash
"""
    similarity_kernel(x, x_dash, std_dist=1.0)

Calculate the distance between the original x and x_dash.

# Arguments
x: The original point.
x_dash: A sample point.
std_dist: The width of the Gaussian kernel (default is 1.0).

# Returns
The distance as a float.
"""
function similarity_kernel(x, x_dash, std_dist=1.0)
    return exp(-norm(x - x_dash)^2 / (std_dist^2))
end


# Sparse linear explanations function
"""
    sparse_linear_explanations(model, x, x_dash, N, K)

Generate a sparse linear explanation for a given model `f` around a point `x`.

# Arguments
model: The model to explain.
x: The original input point.
x_dash: A perturbed version of the input point `x`.
N: Number of samples to generate around `x`.
K: Number of top features to select for the explanation.

# Returns
The top K coefficients of the explanation.
"""
function sparse_linear_explanations(model, x, x_dash, N, K)
    Z_features = Array{Float64}(undef, N, length(x_dash))
    Z_target = Vector{Float64}(undef, N)
    Z_weights = Vector{Float64}(undef, N)

    # Generate N samples at once
    z_dashes = sample_around(x_dash, N) 

    for i in 1:N
        z_dash = z_dashes[i]  # Retrieve the sample
        Z_features[i, :] = z_dash
        Z_target[i] = model(z_dash)
        Z_weights[i] = similarity_kernel(x, z_dash)  # Compute weights inline
    end

    # Apply weights to the feature matrix and target vector (sqrt is applied on the weights because of the ^2 of the distance function)
    Z_features_weighted = Z_features .* sqrt.(Z_weights)
    Z_target_weighted = Z_target .* sqrt.(Z_weights)

    # Create Lasso regression model using GLMNet
    lasso_model = fit(Lasso, Z_features_weighted, Z_target_weighted, λ=0.1)
    
    # Get coefficients path from the fitted model
    coefficients_path = coef(lasso_model)

    # Select the optimal coefficients
    optimal_coefficients = coefficients_path[:, end]

    # Ensure K does not exceed the number of features
    K = min(K, length(optimal_coefficients))

    # Sort the coefficients by their absolute values and select the top K features
    top_k_indices = sortperm(abs.(optimal_coefficients), rev=true)[1:K]

    # Extract the top K coefficients
    top_k_coefs = optimal_coefficients[top_k_indices]

    return top_k_coefs
end


# """
#     times_two(x)

# Multiplies inputs by three
# """
# function times_two(x)
#     return x * 2
# end

# function (method::LIME)(input, output_selector::AbstractOutputSelector)
#     output = method.model(input)                        # y = f(x)
#     output_selection = output_selector(output)          # relevant output
  
# #### Compute VJP at the Points of the output_selector
#     v = zero(output)                                    # vector with zeros
#     v[output_selection] .= 1                            # ones at the relevant indices
#     val = only(back(v))                                 # VJP to get the gradient - v*(dy/dx)
# ###
#     return Explanation(val, output, output_selection, :LIME, :attribution, nothing)
# end
