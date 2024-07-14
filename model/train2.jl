using Flux
using MLDatasets
using Statistics
using Random
using BSON

# Load the FashionMNIST dataset
train_data = FashionMNIST(split=:train)[:]
test_data = FashionMNIST(split=:test)[:]

train_x, train_y = train_data
test_x, test_y = test_data

# Normalize the dataset and reshape
train_x = Float32.(reshape(train_x, 28, 28, 1, :)) ./ 255.0
test_x = Float32.(reshape(test_x, 28, 28, 1, :)) ./ 255.0

# Define the LeNet-5 model
model = Chain(
    Conv((5, 5), 1=>6, relu),   # 1 input channel, 6 output channels
    MaxPool((2, 2)),
    Conv((5, 5), 6=>16, relu),  # 6 input channels, 16 output channels
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256, 120, relu),
    Dense(120, 84, relu),
    Dense(84, 10),              # 10 output classes
    softmax
)

# Define the loss function and optimizer
loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
optimizer = ADAM()

# Convert labels to one-hot encoding
train_y_onehot = Flux.onehotbatch(train_y, 0:9)
test_y_onehot = Flux.onehotbatch(test_y, 0:9)

# Define the data loader
function get_data_loader(x, y, batch_size)
    data_loader = Flux.Data.DataLoader((x, y), batchsize=batch_size, shuffle=true)
    return data_loader
end

# Training function
function train!(model, train_x, train_y, test_x, test_y, epochs, batch_size, optimizer)
    train_loader = get_data_loader(train_x, train_y, batch_size)
    test_loader = get_data_loader(test_x, test_y, batch_size)

    for epoch in 1:epochs
        # Training loop
        for (x, y) in train_loader
            grads = gradient(() -> loss(x, y), Flux.params(model))
            Flux.Optimise.update!(optimizer, Flux.params(model), grads)
        end
        
        # Calculate training and test loss
        train_loss = mean([loss(x, y) for (x, y) in train_loader])
        test_loss = mean([loss(x, y) for (x, y) in test_loader])
        
        println("Epoch: $epoch, Train Loss: $train_loss, Test Loss: $test_loss")
    end
end

# Training the model
epochs = 10
batch_size = 64
train!(model, train_x, train_y_onehot, test_x, test_y_onehot, epochs, batch_size, optimizer)

# 保存模型
# BSON.@save "lenet5_model2.bson" model