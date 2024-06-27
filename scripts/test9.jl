using Random
using LinearAlgebra
using GLM
using DataFrames
using Lasso

# 定义相似度核 π_x（例如指数核）
function similarity_kernel(x, xi, sigma=1.0)
    return exp(-norm(x - xi)^2 / sigma^2)
end

# 定义围绕 x' 采样的函数
function sample_around(x, num_samples=100, perturbation_std=0.1)
    return [x .+ randn(length(x)) * perturbation_std for _ in 1:num_samples]
end

# LIME 算法主函数
function sparse_linear_explanations(f, x, x_prime, N, K)
    Z_features = Array{Float64}(undef, N, length(x_prime))
    Z_target = Vector{Float64}(undef, N)
    Z_weights = Vector{Float64}(undef, N)
    
    for i in 1:N
        z_prime = sample_around(x_prime)[1]
        Z_features[i, :] = z_prime
        Z_target[i] = f(z_prime)
        Z_weights[i] = similarity_kernel(x, z_prime)
    end

    # 对特征矩阵和目标向量应用权重
    Z_features_weighted = Z_features .* sqrt.(Z_weights)
    Z_target_weighted = Z_target .* sqrt.(Z_weights)

    # 创建 Lasso 回归模型
    lasso_model = fit(LassoPath, Z_features_weighted, Z_target_weighted)
    display(lasso_model)

    # 获取所有 λ 下的系数路径
    coefficients_path = coef(lasso_model)

    # 选择 λ 最优解对应的系数
    optimal_coefficients = coefficients_path[:, end]

    # 确保 K 不大于特征数量
    K = min(K, length(optimal_coefficients))

    # 对系数的绝对值进行排序，并选择前 K 个最大的系数对应的特征
    top_k_indices = sortperm(abs.(optimal_coefficients), rev=true)[1:K]

    # 提取前 K 个系数
    top_k_coefs = optimal_coefficients[top_k_indices]

    return top_k_coefs
end

# 示例使用
f = x -> sum(sin, x)  # 示例分类器函数 == sin(x[1]) + sin(x[2]) + sin(x[3])
x = [1.0, 2.0, 3.0]  # 实例
x_prime = [1.0, 2.0, 3.0]  # 实例的可解释版本
N = 100  # 采样数量
K = 3  # 解释长度

w = sparse_linear_explanations(f, x, x_prime, N, K)
println("解释系数: ", w)
