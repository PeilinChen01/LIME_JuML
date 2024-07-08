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
include("superpixel_lime.jl")
include("image_lime.jl")

export LIME
export predict_fn
export batched_image
export calculate_similarity
export weighted_probabilities
export run_lasso_regression
export create_deactivation_matrix
export perturb_image
export plot_labels

# Load and preprocess image
img = load("Image/Clownfish.jpg")
img_permute = permutedims(channelview(img), (3, 2, 1))
img_permute = reshape(img_permute, size(img_permute)..., 1)
input = Float32.(img_permute)

# Define the model
model = ResNet(18; pretrain = true)

probs, target_label = predict_fn(input, 5)

# Felzenszwalb superpixel segmentation
segments = felzenszwalb(img, 1, 200)            # Input als args
superpixel_labels = labels_map(segments)

max_label = maximum(superpixel_labels)

# Generate perturbed images
perturbed_images, deactivated_superpixels = perturb_image(img, superpixel_labels)

# Process perturbed images through the model
probabilities = batched_image(perturbed_images, target_label)

similarities = calculate_similarity(img, perturbed_images)

weighted_probs = weighted_probabilities(probabilities, similarities)

lasso_model = run_lasso_regression(perturbed_images, weighted_probs, deactivated_superpixels)

coef_lasso = Lasso.coef(lasso_model)

optimal_coefs = coef_lasso[:, 10]

important_superpixels = findall(optimal_coefs .!= 0)

plot_labels(important_superpixels, superpixel_labels, img)

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
