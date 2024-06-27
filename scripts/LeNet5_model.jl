using ExplainableAI
using RelevancePropagation
using Flux
using BSON

model = BSON.load("/Users/peilin/.julia/dev/LIME_JuML/src/model.bson", @__MODULE__)[:model] # load pre-trained LeNet-5 model

#=

Chain(
  Conv((5, 5), 1 => 6, relu),           # 156 parameters
  MaxPool((2, 2)),
  Conv((5, 5), 6 => 16, relu),          # 2_416 parameters
  MaxPool((2, 2)),
  Flux.flatten,
  Dense(256 => 120, relu),              # 30_840 parameters
  Dense(120 => 84, relu),               # 10_164 parameters
  Dense(84 => 10),                      # 850 parameters
)                   # Total: 10 arrays, 44_426 parameters, 174.867 KiB.

=#



                
                

