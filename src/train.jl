using Flux
using Flux.Data: DataLoader
using MLDatasets
using Flux: onehotbatch, argmax


device = cpu


# 加载 FashionMNIST 数据集
# Load FashionMNIST dataset
train_x, train_y = FashionMNIST.traindata(Float32)
test_x, test_y = FashionMNIST.testdata(Float32)

# 数据预处理：调整数据维度和标准化
# Data preprocessing: adjusting data dimensions and standardization
train_x = reshape(train_x, 28, 28, 1, 60000) ./ 255.0
test_x = reshape(test_x, 28, 28, 1, 10000) ./ 255.0

# 确保标签是整数
# Ensure that the label is an integer
train_y = Int.(train_y)
test_y = Int.(test_y)

# 检查数据形状
# Check data shape
println("Train X shape: ", size(train_x))
println("Train Y shape: ", size(train_y))
println("Test X shape: ", size(test_x))
println("Test Y shape: ", size(test_y))

# 构建 LeNet-5 模型
# Building the LeNet-5 model
model = Chain(
    Conv((5, 5), 1=>6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6=>16, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 16=>120, relu),  # 调整卷积核大小以适应输入尺寸 Adjusting the size of the convolution kernel to fit the input size
    Flux.flatten,
    Dense(120*2*2, 84, relu),  # 确保全连接层输入尺寸正确 Ensure that the input size of the fully connected layer is correct
    Dense(84, 10),
    softmax
) |> device

# 损失函数和优化器 Loss function and optimizer
loss(x, y) = Flux.crossentropy(model(x), y)
opt = ADAM()

# 将数据转换为适合的格式 Convert data to a suitable format
train_data = DataLoader((train_x, Flux.onehotbatch(train_y, 0:9)), batchsize=64, shuffle=true)
test_data = DataLoader((test_x, Flux.onehotbatch(test_y, 0:9)), batchsize=64)

# 定义 onecold 函数 Define the onecold function
onecold(y) = [argmax(y[:, i]) - 1 for i in 1:size(y, 2)]


# 训练模型 Train the model
for epoch in 1:10
    println("Epoch $epoch")
    for (x, y) in train_data
        x, y = x |> device, y |> device
        gs = Flux.gradient(Flux.params(model)) do
            loss(x, y)
        end
        Flux.Optimise.update!(opt, Flux.params(model), gs)
    end
    println("Training loss: ", loss(train_x |> device, Flux.onehotbatch(train_y, 0:9) |> device))
end

# 模型评估 Evaluation
function accuracy(data)
    correct = 0
    total = 0
    for (x, y) in data
        x, y = x |> device, y |> device
        pred = model(x)
        correct += sum(onecold(pred) .== onecold(y))
        total += size(y, 2)
    end
    return correct / total
end

println("Training accuracy: ", accuracy(train_data))
println("Test accuracy: ", accuracy(test_data))
