using LIME_JuML
using Documenter

DocMeta.setdocmeta!(LIME_JuML, :DocTestSetup, :(using LIME_JuML); recursive=true)

makedocs(;
    modules=[LIME_JuML],
    authors="Peilin Chen <peilin.chen@campus.tu-berlin.de>",
    sitename="LIME_JuML.jl",
    format=Documenter.HTML(;
        canonical="https://PeilinChen01.github.io/LIME_JuML",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => "example.md",
        "Concept" => "concept.md",
        "Important Functions" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/PeilinChen01/LIME_JuML",
    devbranch="main",
)
