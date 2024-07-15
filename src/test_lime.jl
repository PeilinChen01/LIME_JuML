using Random
using LinearAlgebra
using GLM
using DataFrames
using Lasso
using Plots
using Statistics
using MLDatasets
using Images
using ColorTypes
using BSON

# 加载 FashionMNIST 数据集
# load FashionMNIST dataset
function load_FashionMNIST_data()
    train_x, train_y = FashionMNIST.traindata()
    return train_x, train_y
end

# 可视化函数
# create heatmap
function plot_heatmap(image, coefs, title)
    heatmap(image, color=:gray, c=:blues, alpha=0.8, title=title)
end

# 定义相似度核 π_x（例如指数核）
# kernel function
function similarity_kernel(x, xi, sigma=1.0)
    return exp(-norm(x - xi)^2 / sigma^2 )
end

# 定义围绕 x' 采样的函数
function sample_around(x, num_samples=100, perturbation_std=0.1)
    return [x .+ randn(length(x)) * perturbation_std for _ in 1:num_samples]
end

# 生成 λ 序列
function generate_lambda_sequence(X, y; n_lambdas=100)
    λ_max = maximum(abs.(X' * y)) / length(y)
    return λ_max * (0.01 .^ (range(0, stop=1, length=n_lambdas)))
end

# LIME 算法主函数增加可视化
function sparse_linear_explanations_with_heatmap(f, x, x_prime, N, K)
    Z_features = Array{Float64}(undef, N, length(x_prime))
    Z_target = Vector{Float64}(undef, N)
    Z_weights = Vector{Float64}(undef, N)
    
    for i in 1:N
        z_prime = sample_around(x_prime)[1]
        Z_features[i, :] = z_prime
        Z_target[i] = f(z_prime)
        Z_weights[i] = similarity_kernel(x, z_prime)
    end


    # 生成 λ 序列
    λ_seq = generate_lambda_sequence(Z_features, Z_target)

    # 创建 Lasso 回归模型并获取系数路径
    lasso_model = fit(LassoPath, Z_features, Z_target)
    coefficients_path = coef(lasso_model)
    lambda_values = lasso_model.λ

    # 绘制 λ 路径图
    p1 = plot(lambda_values, coefficients_path', xscale=:log10, xlabel="λ (log scale)", ylabel="Coefficients", title="Lasso Path", legend=:bottomleft)
    display(p1)

    # 确保 K 不大于特征数量
    K = min(K, length(coefficients_path[:, end]))

    # 对系数的绝对值进行排序，并选择前 K 个最大的系数对应的特征
    top_k_indices = sortperm(abs.(coefficients_path[:, end][2:end]), rev=true)[1:K]  # 忽略偏置项的系数

    # 提取前 K 个系数
    top_k_coefs = coefficients_path[top_k_indices .+ 1, end]  # 偏移一位以匹配忽略偏置项的索引

    # 显示热图
    image = reshape(x_prime, 28, 28)
    p2 = plot_heatmap(image, top_k_coefs, "Heatmap of FashionMNIST Image with Top K Coefficients")
    display(p2)

    return top_k_coefs
end

# 示例使用

# Load pre-trained model weights (assuming the weights are saved in 'lenet5_model.bson')
#using BSON: @load
# This is for lenet5_model1 @load "lenet5_model2.bson" model
#@load "lenet5_model2.bson" model

# f = model
f = x -> sum(sin.(x) + x.^2)  # 示例非线性分类器函数
train_x, train_y = load_FashionMNIST_data()
x = vec(train_x[:, :, 1])  # 使用第一张图片
x_prime = copy(x)  # 实例的可解释版本
N = 200  # 采样数量
K = 5  # 解释长度


# 确保 K 不大于特征数量
K = min(K, length(x))


w = sparse_linear_explanations_with_heatmap(f, x, x_prime, N, K)
println("For f = sum(sin.(x) + x.^2), first picture in FashionMNIST, num_sample = 200, K = 5, Explanatory coefficient: ", w)

