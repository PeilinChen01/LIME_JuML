using Metalhead: ResNet
using Flux
using Images
using Lasso
using VisionHeatmaps


function predict_fn(input, model, K=1)

    output = model(input)

    probs = softmax(output)

    return probs
end


function batched_image(perturbed_images::Vector, target_index, model, K=1)
    probabilities = []
    for img in perturbed_images
        img = permutedims(channelview(img), (3, 2, 1))
        img = reshape(img, size(img)..., 1)
        input = Float32.(img)
        probs = predict_fn(img, model, K)
        push!(probabilities, probs[target_index])
    end
    return probabilities
end

function calculate_similarity(img, perturbed_images)
    num_perturbed = length(perturbed_images)
    similarities = zeros(num_perturbed)
    for i in 1:num_perturbed
        distance = norm(img .- perturbed_images[i],2)
        similarities[i] = exp(-distance)
    end
    
    return (1 .- similarities)
end

function weighted_probabilities(probabilities, similarities)
    samples = length(probabilities)
    weighted_probs = [probabilities[i] * similarities[i] for i in 1:samples]
    return weighted_probs
end

function run_lasso_regression(weighted_probs, deactivated_superpixels)
    
    X = deactivated_superpixels
    y = weighted_probs
    
    lasso_model = fit(LassoPath, X, y)
    
    return lasso_model
end

function map_coefficients_to_image(superpixel_labels, coefficients)
    height, width = size(superpixel_labels)[1:2]
    coefficient_image = zeros(height, width)
    for label in 1:length(coefficients)
        coefficient_image[superpixel_labels[:,:,1,1] .== label] .= coefficients[label]
    end
    return reshape(transpose(coefficient_image), width, height, 1, 1)
end

function explain_image(input, target_index, model)

        # creating the origanal image
        img_permute_back = reshape(input, size(input)[1:3]...)
        img_original = permutedims(img_permute_back, (3, 2, 1))
        img_original = colorview(RGB, img_original)
        
        # Felzenszwalb superpixel segmentation
        segments = felzenszwalb(img_original, 3, 50)
        superpixel_labels = labels_map(segments)
        
        # Number of superpixels
        max_label = maximum(superpixel_labels)
        
        # Generate perturbed images
        perturbed_images, deactivated_superpixels = perturb_image(img_original, superpixel_labels)

        # Process perturbed images through the model
        probabilities = batched_image(perturbed_images, target_index, model)
        
        # calculate distance
        similarities = calculate_similarity(img_original, perturbed_images)
        
        # weight the data
        weighted_probs = weighted_probabilities(probabilities, similarities)
        
        # create explanations
        lasso_model = run_lasso_regression(weighted_probs, deactivated_superpixels)

        # extract the coefficients
        coef_lasso = Lasso.coef(lasso_model)

        # pixel_relevance
        coef_img = map_coefficients_to_image(superpixel_labels, coef_lasso[:, end])

        # return an explanation
        return coef_img
end