using XAIBase

struct RandomAnalyzer{M} <: AbstractXAIMethod
    model::M
end

function (method::RandomAnalyzer)(input, output_selector::AbstractOutputSelector)
    output = method.model(input)
    output_selection = output_selector(output)

    val = rand(size(input)...)
    return Explanation(val, output, output_selection, :RandomAnalyzer, :sensitivity, nothing)
end

analyzer = RandomAnalyzer(model)
expl = analyze(input, analyzer)

