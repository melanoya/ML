import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score
import seaborn as sns
import subprocess


# make background white again!
sns.set(style="whitegrid", color_codes=True)

# TASK 1
df = pd.read_csv('titanic.csv', index_col='PassengerId')
# print(df.info())

# # на графике изображена доля выживших среди мужчин и женщин
sns.barplot(x="Sex", y="Survived", data=df, palette='Set3')
sns.plt.show()
# # на графике изображена доля выживших среди людей трёх социальных классов
sns.barplot(x="Pclass", y="Survived", data=df, palette='Set3')
sns.plt.show()
# # на графике изображена стоимость билета в зависимости от социально-экономического класса пассажира
sns.barplot(x="Pclass", y="Fare", data=df, palette='Set3')
sns.plt.show()
# из приведенных выше графиков можно сделать вывод, что вероятность выжить у женщин выше (~74%), чем у мужчин (~19%)
# вероятность выжить уменьшается с вместе с классом пасажира. 1 класс ~63%, 2 класс ~49%, 3 класс ~24%.
# стоимость билета пассажиров первого класса в 4 раза дороже, чем стоимость билетов пассажиров третьего класса,
# и почти в пять раз дороже билетов пассажиров третьего класса.

# TASK 2
# на графике изображена средняя вероятность выжить в зависимости от пола и соц. статуса
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df, palette='Set3')
sns.plt.show()
# на основании данных, отраженных на графике, можно сделать вывод, что вероятность выжить у женщины первого класса(~97%)
# и женщины второго класса(~92%) практически не отличается. В свою очередь вероятность выжить у женщины третьего класса
# (~49%) приблизительно в два раза меньше. Вероятность выжить мужчине(~36%) 1 класса в ~2,7 меньше, чем у женщин и
# в 2.25 выше, чем у мужчин второго класса. Вероятность выжить у мужчин второго(~16%) и третьего(~13%) класса
# различается не сильно


# TASK 3
columns = ['Age', 'Sex', 'Pclass', 'Fare']
df['Sex'] = pd.get_dummies(df['Sex'])
df = df.dropna()


# TASK 4
x, y = df[columns], df['Survived']
# print(x.head())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x_test.describe())

clf = DecisionTreeClassifier(min_samples_split=5)
clf.fit(np.array(x_train), np.array(y_train))
importances = pd.Series(clf.feature_importances_, index=columns)
print(importances)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
print(np.mean(cross_val_score(clf, x_train, y_train, cv=5)))

features = list(df['Survived'])


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to produce visualization")

# visualize_tree(clf, features)

# TASK 5
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

scores = []
for t in range(1, 100):
    rfc = RandomForestClassifier(n_estimators=t)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    scores.append(f1_score(y_test, y_pred))
#     rfc.fit(X_train, y_train)
#     y_pred = rfc.predict(X_test)

plt.plot(scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()
