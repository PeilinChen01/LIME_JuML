using Images
using TestImages
using ImageSegmentation
using ColorTypes
using Plots

# Load image
img = load("Image/Clownfish.jpg")
# img = Gray.(testimage("house"))

height, width = size(img)

# Felzenszwalb superpixel segmentation
segments = felzenszwalb(img, 1, 1)  # Input: image, scale parameter, minimum size of superpixels

# Generate superpixel labels
superpixel_labels = labels_map(segments)

# Function to generate the average color of the superpixels
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

# Number of superpixels
max_label = maximum(superpixel_labels)

# Generating the average color of each superpixel
avg_colors = [average_color(img, superpixel_labels, label) for label in 0:max_label]
seg_colored = [avg_colors[label+1] for label in superpixel_labels]

# Plot the superpixels
seg_img = reshape(seg_colored, size(img))
plot(heatmap(seg_img), title="Felzenszwalb Superpixel Segmentation", size=(800, 600))

# Function to perturb the image with deactivated superpixels
function perturb_image(img, superpixels, deactived=100)
    num_superpixels = maximum(superpixels)
    deactived_superpixels = rand(Bool, deactived, num_superpixels)
    perturbed_images = []
    
    for sample in eachrow(deactived_superpixels)
        perturbed_img = copy(img)
        for i in 1:num_superpixels
            if !sample[i]
                perturbed_img[superpixels .== i] .= 0.0
            end
        end
        push!(perturbed_images, perturbed_img)
    end
    
    return perturbed_images, deactived_superpixels
end

# Generate perturbed images
perturbed_images, deactived_superpixels = perturb_image(seg_img, superpixel_labels)

# Plot one of the perturbed images
plot(heatmap(perturbed_images[100]), title="Perturbed Felzenszwalb Superpixel Segmentation", size=(800, 600))

# Function to isolate and plot a superpixel in the original image
function plot_label(label, superpixel_labels, img)
    # Create mask for the chosen superpixel
    mask = superpixel_labels .== label

    # Set all other pixels to black
    highlighted_img = copy(img)
    highlighted_img[.!(mask)] .= RGB(0.0, 0.0, 0.0)

    # Plot the result
    plot(highlighted_img, title="Superpixel Label $label", size=(800, 600))
end

# Example to plot a specific superpixel
label_to_plot = 2
plot_label(label_to_plot, superpixel_labels, img)