using Metalhead: ResNet
using Images
using Flux
using GLMNet
using Distributions
using Lasso


function predict_fn(input, K=1)

    output = model(input)

    probs = softmax(output)

    top_k_indices = sortperm(probs[:,1], rev=true)[1:K]

    # for j in top_k_indices
    #     println("Label: ", labels[j], " mit Wahrscheinlichkeit: ", probs[j])    # 394 ist der Clownfish
    # end

    return probs, top_k_indices[1]
end


function batched_image(perturbed_images::Vector, target_label, K=1)
    probabilities = []
    for img in perturbed_images
        img = permutedims(channelview(img), (3, 2, 1))
        img = reshape(img, size(img)..., 1)
        input = Float32.(img)
        probs, _ = predict_fn(input, K)
        push!(probabilities, probs[target_label])
    end
    return probabilities
end

function calculate_similarity(img, perturbed_images)
    img_vector = vec(channelview(img))
    num_perturbed = length(perturbed_images)
    
    similarities = zeros(num_perturbed)
    for i in 1:num_perturbed
        perturbed_vector = vec(channelview(perturbed_images[i]))
        # L2-Norm distance
        similarities[i] = norm(img_vector - perturbed_vector, 2)
    end
    # # calculate the similarities similar images get higher values
    # similarities = exp.(-similarities)
    
    return similarities
end

function weighted_probabilities(probabilities, similarities)
    samples = length(probabilities)
    weighted_probs = [probabilities[i] * similarities[i] for i in 1:samples]
    return weighted_probs
end

function run_lasso_regression(perturbed_images, weighted_probs, deactivated_superpixels, alpha=1.0)
    num_samples = length(deactivated_superpixels)
    num_superpixels = maximum(vcat(deactivated_superpixels...))
    
    X = zeros(num_samples, num_superpixels)
    y = weighted_probs
    
    for i in 1:num_samples
        for j in deactivated_superpixels[i]
            X[i, j] = 1.0
        end
    end

    lasso_model = fit(LassoPath, X, y)
    
    return lasso_model
end