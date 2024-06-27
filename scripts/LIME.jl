using XAIBase
using MLJ
using DataFrames
using Random
using Plots

# Load example data
X, y = @load_iris
df = DataFrame(X)
target = y

# Train a simple model
model = @load DecisionTreeClassifier
mach = machine(model, df, target)
fit!(mach)

# Create LIME explainer
explainer = lime_explainer(df, mach)

# Explain a specific instance
instance = df[1, :]
explanation = explain(explainer, instance)

# Generate and visualize heatmap
heatmap = heatmap(explanation)
plot(heatmap, title="LIME Explanation Heatmap")



