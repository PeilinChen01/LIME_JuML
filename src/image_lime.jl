using Metalhead: ResNet
using Flux
using Images
using Lasso
using VisionHeatmaps

"""
    predict_fn(input, model, K=1)

Generate prediction probabilities for a given input using the provided model.

# Arguments
- `input`: The input data for which predictions are to be generated.
- `model`: The machine learning model used to make predictions. The model should be callable with the input data.
- `K`: An optional parameter (default is 1). It represents the top K predictions to consider. Currently not used in the function but can be incorporated for future enhancements.

# Returns
- `probs`: The prediction probabilities for the given input, obtained by applying the softmax function to the model's output.

# Description
This function takes an input and a model, generates predictions using the model, and then applies the softmax function to obtain the prediction probabilities. The softmax function ensures that the output probabilities sum to 1, making it suitable for classification tasks.

# Example
```julia
# Define a simple model
model(x) = x * 2  # A dummy model for illustration

# Define an input
input = [1.0, 2.0, 3.0]

# Generate prediction probabilities
probs = predict_fn(input, model)
```
"""
function predict_fn(input, model, K=1)

    output = model(input)

    probs = softmax(output)

    return probs
end

"""
    batched_image(perturbed_images::Vector, target_index, model, K=1)

Generate prediction probabilities for a batch of perturbed images using the provided model.

# Arguments
- `perturbed_images::Vector`: A vector containing perturbed versions of the original image. Each image in the vector should be processed individually.
- `target_index`: The index of the target class for which the prediction probabilities are to be extracted.
- `model`: The machine learning model used to make predictions. The model should be callable with the input data.
- `K`: An optional parameter (default is 1). It represents the top K predictions to consider. Currently not used in the function but can be incorporated for future enhancements.

# Returns
- `probabilities`: A vector containing the prediction probabilities for the specified target class for each perturbed image.

# Description
This function processes a batch of perturbed images, generates predictions for each image using the provided model, and extracts the prediction probabilities for the specified target class. The function performs the following steps:
1. Permute the dimensions of each image to match the model's expected input format.
2. Reshape the image to ensure compatibility with the model.
3. Convert the image to a `Float32` array.
4. Generate prediction probabilities using the `predict_fn` function.
5. Extract the prediction probability for the specified target class and store it in a vector.

# Example
```julia
# Define a simple model
model(x) = x * 2  # A dummy model for illustration

# Create a vector of perturbed images (dummy images for illustration)
perturbed_images = [rand(3, 224, 224) for _ in 1:10]

# Define the target index
target_index = 1

# Generate prediction probabilities for the perturbed images
probs = batched_image(perturbed_images, target_index, model)
```
"""
function batched_image(perturbed_images::Vector, target_index::Int, model, K=1)
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

"""
    calculate_similarity(img, perturbed_images::Vector)

Calculate the similarity between the original image and each perturbed image using the L2 norm.

# Arguments
- `img`: The original image. It should be in the same format as the perturbed images.
- `perturbed_images::Vector`: A vector containing perturbed versions of the original image.

# Returns
- `similarities`: A vector containing the similarity values for each perturbed image, where higher values indicate greater similarity to the original image.

# Description
This function calculates the similarity between the original image and each perturbed image. The similarity is computed using the L2 norm (Euclidean distance) between the images. The L2 norm is calculated for each perturbed image with respect to the original image. The similarity is then computed as the exponential of the negative distance. Finally, the similarity values are inverted (1 - similarity) to emphasize higher similarity values for closer distances.

# Example
```julia
# Define the original image (dummy image for illustration)
original_image = rand(224, 224, 3)

# Create a vector of perturbed images (dummy images for illustration)
perturbed_images = [rand(224, 224, 3) for _ in 1:10]

# Calculate similarities
similarities = calculate_similarity(original_image, perturbed_images)
```
"""
function calculate_similarity(img, perturbed_images::Vector)
    num_perturbed = length(perturbed_images)
    similarities = zeros(num_perturbed)
    for i in 1:num_perturbed
        distance = norm(img .- perturbed_images[i],2)
        similarities[i] = exp(-distance)
    end
    
    return (1 .- similarities)
end

"""
    weighted_probabilities(probabilities::Vector, similarities::Vector)

Calculate the weighted probabilities by multiplying the probabilities by the similarities.

# Arguments
- `probabilities::Vector`: A vector of probabilities predicted by the model for each perturbed image.
- `similarities::Vector`: A vector of similarity values between the original image and each perturbed image.

# Returns
- `weighted_probs`: A vector containing the weighted probabilities, where each probability has been scaled by its similarity to the original image.

# Description
This function computes the weighted probabilities for each perturbed image. Each probability is multiplied by its corresponding similarity value, which is a measure of how similar the perturbed image is to the original image. The resulting weighted probabilities give more importance to perturbed images that are more similar to the original image.

The point-wise multiplication of the `probabilities` vector and the `similarities` vector is performed element-wise.

# Example
```julia
# Define the predicted probabilities for perturbed images (dummy values for illustration)
probabilities = [0.1, 0.4, 0.3, 0.2]

# Define the similarity values for perturbed images (dummy values for illustration)
similarities = [0.9, 0.8, 0.7, 0.6]

# Calculate weighted probabilities
weighted_probs = weighted_probabilities(probabilities, similarities)

# `weighted_probs` will be [0.09, 0.32, 0.21, 0.12]
"""
function weighted_probabilities(probabilities::Vector, similarities::Vector)
    weighted_probs = probabilities.*similarities
    return weighted_probs
end

"""
    run_lasso_regression(weighted_probs::Vector, deactivated_superpixels::BitMatrix)

Perform Lasso regression to fit a model using weighted probabilities and deactivated superpixels.

# Arguments
- `weighted_probs::Vector`: A vector of weighted probabilities for the perturbed images. Each probability is multiplied by the similarity between the perturbed image and the original image.
- `deactivated_superpixels::BitMatrix`: A binary matrix where each row represents a perturbed image and each column represents a superpixel. The value is `true` if the superpixel is deactivated in that perturbed image and `false` otherwise.

# Description
This function performs Lasso regression to learn a sparse linear model that explains the relationship between the perturbed images and their corresponding weighted probabilities. The Lasso regression will be used to select the most important superpixels by applying L1 regularization.

- `X` is the design matrix containing binary features indicating which superpixels are deactivated for each perturbed image.
- `y` is the vector of weighted probabilities associated with each perturbed image.
- The `Lasso` function from the `Lasso` package is used to fit the Lasso regression model, which performs L1 regularization to select a subset of features (superpixels) that best explain the weighted probabilities.

# Returns
- `lasso_model::Lasso`: The fitted Lasso regression model. The model contains the coefficients for each superpixel, which can be used to understand the contribution of each superpixel to the explanation of the prediction.
"""
function run_lasso_regression(weighted_probs::Vector, deactivated_superpixels::BitMatrix)
    
    X = deactivated_superpixels
    y = weighted_probs
    
    lasso_model = fit(LassoPath, X, y)
    
    return lasso_model
end
"""
    map_coefficients_to_image(superpixel_labels, coefficients)

Maps Lasso regression coefficients onto the superpixels of the original image to visualize the importance of each superpixel for the target class.

# Arguments

- `superpixel_labels::Matrix`: An matrix where each entry represents the superpixel label for a pixel in the image.

- `coefficients`: It contains the coefficients from the Lasso regression model for each superpixel.

### Returns

- `coef_img`: Values representing the importance of each superpixel.

### Description

The `map_coefficients_to_image` function generates a visual representation of the importance of each superpixel based on the Lasso regression coefficients. The coefficients reflect how much each superpixel contributes to the model's prediction for the target class. This function creates an image where each pixel's value is determined by the coefficient of the superpixel to which it belongs, making it easier to visualize which superpixels are most influential for the prediction.
"""
function map_coefficients_to_image(superpixel_labels::Matrix, coefficients)
    height, width = size(superpixel_labels)[1:2]
    coefficient_image = zeros(height, width)
    for label in 1:length(coefficients)
        coefficient_image[superpixel_labels[:,:,1,1] .== label] .= coefficients[label]
    end
    return reshape(transpose(coefficient_image), width, height, 1, 1)
end

"""
    explain_image(input, target_index, model)

Generates a visual explanation for a given image using Lasso regression and superpixel segmentation to highlight the importance of different superpixels.

# Arguments
- `input::Array{Float32, 4}`: The input image tensor with shape (C, H, W, N), where C is the number of color channels (3 for RGB), H is the height, W is the width, and N is the batch size. The tensor is assumed to be in (Channel, Height, Width) format.
- `target_index::Int`: The index of the target class for which the explanation is to be generated. This is used to extract the probabilities associated with the target class.
- `model::AbstractModel`: The trained model used for making predictions on the perturbed images. The model should be compatible with the `batched_image` function to process images and extract probabilities.

# Returns
- `coef_img::Array{Float32, 2}`: A 2D array representing the importance of each superpixel for the target class. This image can be visualized to understand which parts of the image are more significant for the model's prediction.

# Description
This function creates a visual explanation of the image by performing the following steps:

1. **Image Preprocessing**:
   - Reshapes the input tensor and converts it to a `RGB` image.
   
2. **Superpixel Segmentation**:
   - Applies Felzenszwalb's algorithm to segment the image into superpixels.

3. **Generate Perturbed Images**:
   - Creates perturbed versions of the image by deactivating different superpixels.

4. **Process Perturbed Images**:
   - Feeds the perturbed images through the model to obtain class probabilities.

5. **Calculate Similarity**:
   - Computes the similarity between the original and perturbed images.

6. **Weight Data**:
   - Weighs the probabilities based on the similarity measures.

7. **Run Lasso Regression**:
   - Performs Lasso regression to determine the importance of each superpixel for the target class.

8. **Extract Coefficients**:
   - Extracts the coefficients from the Lasso model to assess the relevance of each superpixel.

9. **Map Coefficients to Image**:
   - Maps the coefficients to the superpixel labels to generate a visual explanation.
"""
function explain_image(input::Array{Float32, 2}, target_index::Int, model::AbstractModel)

        # creating the origanal image
        img_permute_back = reshape(input, size(input)[1:3]...)
        img_original = permutedims(img_permute_back, (3, 2, 1))
        img_original = colorview(RGB, img_original)
        
        # Felzenszwalb superpixel segmentation
        segments = felzenszwalb(img_original, 1, 200)
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
        coef_img = map_coefficients_to_image(superpixel_labels, coef_lasso[:, 10])

        # return an explanation
        return coef_img
end