import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# Set global styles for plots
sns.set_style(style='white')
sns.set_context(context='notebook', font_scale=1.3, rc={'figure.figsize': (16, 9)})

df = pd.read_csv('data/sample.csv')
df.head()

df = df[df.sentiment != 0]
counts = df.sentiment.value_counts()
print(counts)

print("\nPredicting only -1 = {:.2f}% accuracy".format(counts[-1] / sum(counts) * 100))

X = df.text
y = df.sentiment

print(X.shape)
print(y.shape)

vector = CountVectorizer(max_features=1000, binary=True)

sm = SMOTE()

models = [
    ("MultinomialNB", MultinomialNB()),
    ("BernoulliNB", BernoulliNB()),
    ("LogisticRegression", LogisticRegression()),
    ("SGDClassifier", SGDClassifier()),
    ("LinearSVC", LinearSVC()),
    ("RandomForestClassifier", RandomForestClassifier()),
    ("MLPClassifier", MLPClassifier())
]


def benchmark(model, X, y, n):
    scores = []
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.33, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_vect = vector.fit_transform(X_train)
        X_test_vect = vector.transform(X_test)

        X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)

        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test_vect)

        acc = accuracy_score(y_test, y_pred)

        scores.append(acc)
    return np.mean(scores)


train_sizes = [1000, 5000, 10000, 15000, 20000]
table = []
for name, model in models:
    for n in train_sizes:
        table.append({
            'model': name,
            'accuracy': benchmark(model, X, y, n),
            'train_size': n
        })

df = pd.DataFrame(table)

plt.figure(figsize=(16, 9))
fig = sns.pointplot(x='train_size', y='accuracy', hue='model',
                    data=df[df.model.map(
                        lambda x: x in ["MultinomialNB", "BernoulliNB", "LogisticRegression", "SGDClassifier",
                                        "LinearSVC", "RandomForestClassifier", "MLPClassifier"])])
sns.set_context("notebook", font_scale=1.5)
fig.set(ylabel="Accuracy")
fig.set(xlabel="Training set size")
fig.set(title="Other Classification Algorithms")
plt.show()
