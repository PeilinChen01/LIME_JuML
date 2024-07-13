# lime implementation
using LinearAlgebra
using XAIBase
"""
    LIME(model)

# Arguments
- `model::M`: A machine learning model which is used to make predictions. The model should be callable with input data and return output predictions.

# Description
The `LIME` (Local Interpretable Model-agnostic Explanations) struct creates an analyzer for explaining the predictions of a machine learning model. LIME aims to provide interpretable explanations for the predictions made by black-box models by approximating the model locally with an interpretable model.

# Usage


"""
struct LIME{M} <: AbstractXAIMethod
    model::M
end

function (method::LIME)(input, output_selector::AbstractOutputSelector)
    output = method.model(input)                        
    output_selection = output_selector(output)          

    val = explain_image(input, output_selection[1], method.model)
    extras = nothing
    return Explanation(val, output, output_selection, :LIME, :attribution, extras)
end

# # Load and preprocess image
# img = load("Image/Clownfish.jpg")
# img_permute = permutedims(channelview(img), (3, 2, 1))
# img_permute = reshape(img_permute, size(img_permute)..., 1)
# input = Float32.(img_permute)

# # Define the model
# model = ResNet(18; pretrain = true)

# analyzer = LIME(model)

# expl = analyze(input, analyzer)

# probs, target_index = predict_fn(input, model)

# # Felzenszwalb superpixel segmentation
# segments = felzenszwalb(img, 3, 200)            # Input als args
# superpixel_labels = labels_map(segments)

# # Number of superpixels
# max_label = maximum(superpixel_labels)

# # Generate perturbed images
# perturbed_images, deactivated_superpixels = perturb_image(img, superpixel_labels)

# # Process perturbed images through the model
# probabilities = batched_image(perturbed_images, target_index, model)

# similarities = calculate_similarity(img, perturbed_images)

# weighted_probs = weighted_probabilities(probabilities, similarities)

# lasso_model = run_lasso_regression(perturbed_images, weighted_probs, deactivated_superpixels)

# coef_lasso = Lasso.coef(lasso_model)

# optimal_coefs = coef_lasso[:, 10]

# important_superpixels = findall(optimal_coefs .!= 0)

# plot_labels(important_superpixels, superpixel_labels, img)