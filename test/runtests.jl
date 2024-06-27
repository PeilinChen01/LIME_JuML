using LIME_JuML
using Test
using XAIBase

@testset "LIME_JuML.jl" begin
    # Write your tests here.
    @testset "times two tests" begin
        @test times_two(2.0) == 4.0
        @test times_two(4) == 8
        @test times_two(5) == 10
    end
end
