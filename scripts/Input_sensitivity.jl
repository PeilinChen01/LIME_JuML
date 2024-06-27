using XAIBase
using Zygote: gradient

struct MyGradient{M} <: AbstractXAIMethod
    model::M
end

function (method::MyGradient)(input, output_selector::AbstractOutputSelector)
    output = method.model(input)
    output_selection = output_selector(output)

    grad = gradient((x) -> only(method.model(x)[output_selection]), input)
    val = only(grad)
    return Explanation(val, output, output_selection, :MyGradient, :sensitivity, nothing)
end

