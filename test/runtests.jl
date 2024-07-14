using LIME_JuML
using Test
using XAIBase

@testset "LIME_JuML.jl" begin
    # Write your tests here.
    @testset "sample around test" begin
        @test sample_around([1; 1], 1) == [[0.878584298512076, 1.0872814156171304]]
    end
    @testset "similarity kernel test" begin
        @test similarity_kernel([1.0, 1.0], [1.1, 1.1], 1.0) == 0.9801986733067553
    end
    @testset "sparse linear explanations test" begin
        @test sparse_linear_explanations(x -> sum(x), [1.0, 1.0], [1.1, 1.1], 10, 1)[1] == 0.9706216694190505
    end
end
