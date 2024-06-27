using Random
using LinearAlgebra
using GLM
using DataFrames
using Lasso

# 构建特征矩阵和目标向量
X = rand(100, 10)  # 100 个样本，10 个特征
y = rand(100)      # 100 个目标值

# 运行 Lasso 回归
lasso_model = fit(LassoPath, X, y)

# 获取所有 λ 下的系数路径
coefficients_path = coef(lasso_model)

# 定义要选择的特征数量 K
K = 3

# 选择 λ 最优解对应的系数
optimal_coefficients = coefficients_path[:, end]

# 对系数的绝对值进行排序，并选择前 K 个最大的系数对应的特征
top_k_indices = sortperm(abs.(optimal_coefficients), rev=true)[1:K]

# 输出选择的特征索引
println("选择的前 $K 个特征索引: ", top_k_indices)
