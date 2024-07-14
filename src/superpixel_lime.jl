using Images
using Random
using ImageSegmentation
using ColorTypes

"""
    average_color(img, labels, label)

Calculate the average color of a specific superpixel region in an image.

# Arguments
- `img`: The input image.
- `labels`: The label matrix where each pixel is labeled with its superpixel ID.
- `label`: The specific label (superpixel ID) for which the average color is to be calculated.

# Returns
- If the image is grayscale, returns the average gray value as an RGB color.
- If the image is colored, returns the average RGB color for the specified superpixel.

# Description
This function calculates the average color of a given superpixel region within an image. For grayscale images, it computes the average gray value and converts it to an RGB color. For colored images, it calculates the average red, green, and blue values.

"""

function average_color(img, labels, label)
    if eltype(img) <: Gray
        total = 0.0
        count = 0
        for i in 1:size(img, 1)
            for j in 1:size(img, 2)
                if labels[i, j] == label
                    total += Float64(img[i, j])
                    count += 1
                end
            end
        end
        gray_value = total / count
        return RGB(gray_value, gray_value, gray_value)
    else
        r_total = 0.0
        g_total = 0.0
        b_total = 0.0
        count = 0
        for i in 1:size(img, 1)
            for j in 1:size(img, 2)
                if labels[i, j] == label
                    color = img[i, j]
                    r_total += red(color)
                    g_total += green(color)
                    b_total += blue(color)
                    count += 1
                end
            end
        end
        return RGB(r_total / count, g_total / count, b_total / count)
    end
end

"""
    create_deactivation_matrix(num_superpixels, samples, threshold=0.03)

Create a deactivation matrix for superpixels.

# Arguments
- `num_superpixels`: The number of superpixels in the image.
- `samples`: The number of perturbed image samples to generate.
- `threshold`: The probability threshold for deactivating a superpixel (default is 0.03).

# Returns
- `deactivated_matrix`: A binary matrix of size (samples, num_superpixels) where each entry indicates whether a superpixel is deactivated (1) or not (0).

# Description
This function generates a deactivation matrix that specifies which superpixels should be deactivated in each sample. The deactivation process is probabilistic, with each superpixel having a chance of being deactivated based on the specified threshold. The function ensures that each superpixel is deactivated in at least one sample.

The process involves:
1. Creating a random matrix of size (samples, num_superpixels) with values between 0 and 1.
2. Setting values below the threshold to 1 (deactivated) and values above the threshold to 0 (not deactivated).
3. Ensuring that each superpixel is deactivated in at least one sample by checking each superpixel column and setting at least one entry to 1 if none are already.

# Example
```julia
# Create a deactivation matrix for 50 superpixels and 100 samples
deactivated_matrix = create_deactivation_matrix(50, 100)

# Print the deactivation matrix
println(deactivated_matrix)
"""

function create_deactivation_matrix(num_superpixels, samples, threshold=0.03)
    # creates a matrix with numbers between 0 and 1
    rand_matrix = rand(samples, num_superpixels)
    
    # all numbers under the threshold will be set to 1, the rest will be set to 0
    deactivated_matrix = rand_matrix .< threshold
    
    # checks deactivation of superpixels
    for superpixel in 1:num_superpixels
        if !any(deactivated_matrix[:, superpixel])
            sample_index = rand(1:samples)
            deactivated_matrix[sample_index, superpixel] = true
        end
    end

    return deactivated_matrix
end

"""
    perturb_image(img, superpixels, samples=100)

Generate perturbed images by deactivating superpixels.

# Arguments
- `img`: The input image to perturb. The image can be in grayscale or color.
- `superpixels`: The label matrix where each pixel is labeled with its superpixel ID.
- `samples`: The number of perturbed image samples to generate (default is 100).

# Returns
- `perturbed_images`: A vector of perturbed images where specific superpixels are deactivated.
- `deactivated_superpixels`: A binary matrix indicating which superpixels were deactivated in each sample.

# Description
This function generates perturbed versions of the input image by deactivating (setting to 0) superpixels according to a deactivation matrix. Each perturbed image is created by copying the original image and deactivating the superpixels as specified. The function ensures that each superpixel is deactivated in at least one sample.

The deactivation process involves:
1. Creating a deactivation matrix using `create_deactivation_matrix` where entries indicate whether a superpixel is deactivated.
2. Iterating through the number of samples and for each sample:
   - Copying the original image.
   - Deactivating the superpixels specified by the deactivation matrix by setting the pixel values to 0.
   - Storing the perturbed image in a vector.

# Example
```julia
using Images, ImageSegmentation

"""

function perturb_image(img, superpixels, samples=100)
    num_superpixels = maximum(superpixels)
    deactivated_superpixels = create_deactivation_matrix(num_superpixels, samples)
    perturbed_images = []
    for i in 1:samples
        perturbed_img = copy(img)
        for id in 1:num_superpixels
            if deactivated_superpixels[i, id] == 1
                perturbed_img[superpixels .== id] .= 0.0
            end
        end
        push!(perturbed_images, perturbed_img)
    end
    return perturbed_images, deactivated_superpixels
end

"""
    plot_labels(labels, superpixel_labels, img)

Highlight and plot selected superpixels on an image.

# Arguments
- `labels`: A vector of superpixel labels to highlight.
- `superpixel_labels`: A matrix of the same size as the image, where each entry represents the label of the superpixel.
- `img`: The original image to highlight superpixels on.

# Description
This function creates a highlighted version of the input image by setting all pixels outside the selected superpixels to black. It then plots the highlighted image.

The process involves:
1. Creating a mask for the chosen superpixels.
2. Setting all other pixels in the image to black.
3. Plotting the result.
"""

function plot_labels(labels, superpixel_labels, img)
    # Create a mask for the chosen superpixels
    mask = zeros(Bool, size(superpixel_labels))
    for label in labels
        mask .|= (superpixel_labels .== label)
    end

    # Set all other pixels to black
    highlighted_img = copy(img)
    highlighted_img[.!mask] .= RGB(0.0, 0.0, 0.0)

    # Plot the result
    plot(highlighted_img, size=(800, 600))
end

# # Example to plot a specific superpixel
# label_to_plot = 1
# plot_labels(label_to_plot, superpixel_labels, img)

# # Load image
# img = load("Image/Clownfish.jpg")
# img = Gray.(testimage("house"))

# height, width = size(img)

# # Felzenszwalb superpixel segmentation
# segments = felzenszwalb(img, 1, 200)  # Input: image, scale parameter, minimum size of superpixels

# # Generate superpixel labels
# superpixel_labels = labels_map(segments)

# # Number of superpixels
# max_label = maximum(superpixel_labels)

# # Generating the average color of each superpixel
# avg_colors = [average_color(img, superpixel_labels, label) for label in 0:max_label]
# seg_colored = [avg_colors[label+1] for label in superpixel_labels]

# # Plot the superpixels
# seg_img = reshape(seg_colored, size(img))
# plot(heatmap(seg_img), title="Felzenszwalb Superpixel Segmentation", size=(800, 600))
