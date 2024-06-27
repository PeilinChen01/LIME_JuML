using DataFrames
using LinearAlgebra
using Distributions
using StatsBase

# 假设复杂模型是一个函数
complex_model(x) = x[1] * 2 + x[2] * 3 + rand(Normal(0, 0.1))

# 生成扰动样本
function generate_perturbations(instance::Vector{Float64}, num_samples::Int)
    perturbations = [instance .+ rand(Normal(0, 0.1), length(instance)) for _ in 1:num_samples]
    return hcat(perturbations...)
end

# LIME算法
function lime(instance::Vector{Float64}, model::Function, num_samples::Int=100)
    # 生成扰动样本
    perturbations = generate_perturbations(instance, num_samples)
    
    # 获取复杂模型的预测
    predictions = [model(perturbation) for perturbation in eachrow(perturbations)]
    
    # 计算距离（权重）
    distances = [norm(instance - perturbation) for perturbation in eachrow(perturbations)]
    weights = exp.(-distances .^ 2)
    
    # 拟合线性模型
    X = hcat(ones(num_samples), perturbations)
    W = Diagonal(weights)
    β = (X * W * X) \ (X * W * predictions)
    
    return β
end

# 示例
instance = [1.0, 2.0]
β = lime(instance, complex_model)
println("解释模型的系数: ", β)
