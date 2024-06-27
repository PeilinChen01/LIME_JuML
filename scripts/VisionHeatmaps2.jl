using VisionHeatmaps

analyzer = MyGradient(model)
expl = analyze(input, analyzer)
heatmap(expl.val)

