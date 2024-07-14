# LIME_JuML

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://PeilinChen01.github.io/LIME_JuML/dev/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://PeilinChen01.github.io/LIME_JuML/dev/)
[![Build Status](https://github.com/PeilinChen01/LIME_JuML/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/PeilinChen01/LIME_JuML/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/PeilinChen01/LIME_JuML/branch/main/graph/badge.svg)](https://codecov.io/gh/PeilinChen01/LIME_JuML)

The LIME_JuML package implements the explainable AI methods LIME for image inputs. Using XAIBase as structurell guildline. You can read more about XAIBase [here](https://julia-xai.github.io/XAIBase.jl/dev) .

[Here](https://PeilinChen01.github.io/LIME_JuML/dev/) you can find the documentation for LIME_JuML.


## Step by Step Installation
Here are the steps to install this pac:

1. Open the Julia command-line interface.

2.  Enter the Pkg mode by pressing `]`.

3.  Run the following command to add Lime:

```julia
add https://github.com/PeilinChen01/LIME_JuML.jl
```

## Panda Image Example
```julia
using XAIBase
using VisionHeatmaps
using Metalhead: ResNet
using LIME_JuML
using Images

img = load("Image/panda.jpg")

display(img)
img = permutedims(channelview(img),(3,2,1))
img = reshape(img, size(img)..., 1)
input = Float32.(img)

model = ResNet(18; pretrain = true);
analyzer = LIME(model)
expl = analyze(input, analyzer);

heat = VisionHeatmaps.heatmap(expl.val)
display(heat)
```
