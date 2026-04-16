import plotly.graph_objects as go
import json

# Carica le accuracy salvate da train.py
with open("models_accuracy.json", "r") as f:
    models_accuracy = json.load(f)

tags = ('fnn', 'cnn', 'scrambled_cnn', 'scrambled_fnn')
accuracy_list = [models_accuracy[tag] for tag in tags]

fig = go.Figure([go.Bar(x=tags, y=accuracy_list)])
fig.update_layout(title='Performance comparison',
                  yaxis_title="Accuracy [%]",
                  xaxis_title="Model type",
                  width=700,
                  height=350)
fig.show()