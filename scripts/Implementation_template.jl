struct MyMethod{M} <: AbstractXAIMethod 
    model::M    
end

function (method::MyMethod)(input, output_selector::AbstractOutputSelector)
    output = method.model(input)
    output_selection = output_selector(output)

    val = ...         # your method's implementation
    extras = nothing  # optionally add additional information using a named tuple
    return Explanation(val, output, output_selection, :MyMethod, :attribution, extras)
end