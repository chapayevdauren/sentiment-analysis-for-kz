import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style(style='white')
sns.set_context(context='notebook', font_scale=1.3, rc={'figure.figsize': (16, 9)})

acc = [62.93, 56.09, 58.32, 64.96, 63.83]
train_sizes = [1000, 5000, 10000, 15000, 20000]
table = []

for index, item in enumerate(train_sizes):
    table.append({
        'model': "LSTM + Word2Vec",
        'accuracy': acc[index],
        'train_size': item
    })

df = pd.DataFrame(table)

plt.figure(figsize=(15, 6))
fig = sns.pointplot(x='train_size', y='accuracy', hue='model',
                    data=df[df.model.map(lambda x: x in ["LSTM + Word2Vec"])])
sns.set_context("notebook", font_scale=1.5)
fig.set(ylabel="Accuracy")
fig.set(xlabel="Training set size")
fig.set(title="LSTM + Word2Vec Benchmark")
plt.show()
