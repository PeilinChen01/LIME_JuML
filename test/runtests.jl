using LIME_JuML
using Test

@testset "LIME_JuML.jl" begin
    # Write your tests here.
    @testset "superpixel test" begin
        
        img = load("Image/test.jpg")
        
        segments = felzenszwalb(img, 1, 20)
        superpixel_labels = labels_map(segments)
        max_label = maximum(superpixel_labels)

        @test perturbed_images, deactivated_superpixels = perturb_image(img, superpixel_labels, 2)
    end
end
