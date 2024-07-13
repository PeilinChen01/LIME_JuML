using Images
using Random
using ImageSegmentation
using ColorTypes

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

# Function to perturb the image with deactivated superpixels
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
