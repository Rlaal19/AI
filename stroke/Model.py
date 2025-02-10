import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt


# load dataset
data = pd.read_csv("/Users/parichaya23icloud.com/Desktop/AI/Miniproject_Model/cleandata_onehot_and_label_encoding.csv", header=0)
X = data.drop('Heart_Attack_Risk', axis=1)
y = data['Heart_Attack_Risk'] # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

def logistic_model(C, solver_, multiclass_):
    logistic_regression_model = LogisticRegression(random_state=42, solver=solver_, multi_class=multiclass_, n_jobs=1, C=C)
    return logistic_regression_model

multiclass = ['ovr', 'multinomial']
solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
scores = []
params = []

for i in multiclass:
    for j in solver_list:
        try:
            model = logistic_model(1, j, i)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            params.append(i + ' ' + j)
            accuracy = accuracy_score(y_test, predictions)
            scores.append(accuracy)
        except:
            None

sns.barplot(x=params, y=scores).set_title('Beans Accuracy')
plt.xticks(rotation=90)

model = logistic_model(1, 'newton-cg', 'multinomial')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))


# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()



    