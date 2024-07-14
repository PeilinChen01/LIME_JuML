module LIME_JuML

# Write your package code here.
include("lime.jl")
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
export explain_image

end
