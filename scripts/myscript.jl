using XAIBase
using Zygote: gradient
using VisionHeatmaps

using ExplainableAI
using RelevancePropagation
using Flux
using BSON

##
model = BSON.load("/Users/peilin/.julia/dev/LIME_JuML/src/model.bson", @__MODULE__)[:model] # load pre-trained LeNet-5 model

##
struct MyGradient{M} <: AbstractXAIMethod
    model::M
end

##
function (method::MyGradient)(input, output_selector::AbstractOutputSelector)
    output = method.model(input)
    output_selection = output_selector(output)

    grad = gradient((x) -> only(method.model(x)[output_selection]), input)
    val = only(grad)
    return Explanation(val, output, output_selection, :MyGradient, :sensitivity, nothing)
end

##
analyzer = MyGradient(model)

##
expl = analyze(input, analyzer)

##
heatmap(expl.val)




