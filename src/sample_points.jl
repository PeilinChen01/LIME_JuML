using Images
using Plots
using ColorTypes
using Random

# function to sample around the instance x
function get_sample_points(img, center, num_samples=20)
    x, y = center
    height, width = size(img)
    range_x = floor(width/4)
    range_y = floor(height/4)

    # generates random sample-coordinaten
    dx = rand(-range_x:range_x, num_samples)
    dy = rand(-range_y:range_y, num_samples)

    # samples are clamped to the image
    new_x = clamp.(x .+ dx, 1, height)
    new_y = clamp.(y .+ dy, 1, width)

    # calculate the distance of the samples to the instance x
    distances = sqrt.((new_x .- x).^2 .+ (new_y .- y).^2)

    return new_x, new_y, distances
end

### Example:

# load Image
img = load("Image/Clownfish.jpg")

# define center of samples
instance_x = (100, 150)  # instance x

# getting the sample points
x_samples, y_samples, distances = get_sample_points(img, instance_x)

# sizes of the marker for the scatterplot
sizes = 10 .+ 40 .* exp.(- 0.05 .*distances)

# plot the image
plot(
    img, legend=false, xlims=(0, size(img, 2)), ylims=(0, size(img, 1)), size=(800, 600)
)

# scatter-plot of the sample points
scatter!(
    x_samples, y_samples, markersize=sizes, alpha=0.7
)

# set the marker of the instance x
scatter!(
    [instance_x[1]],[instance_x[2]], marker=:x, markersize=20, color=:black
)