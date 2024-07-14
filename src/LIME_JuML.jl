module LIME_JuML

# Write your package code here.
include("superpixel_lime.jl")
include("image_lime.jl")
include("lime.jl")
include("lenet5_lime.jl")

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
export sample_around_image

end
