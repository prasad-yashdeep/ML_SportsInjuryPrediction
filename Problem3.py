from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from Problem1 import clean_weather
from Problem1 import clean_stadiumtype
from Problem1 import clean_play_df
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support


play_df = pd.read_csv('Dataset/PlayList.csv')
play_df_cleaned = clean_play_df(play_df)
injury_df = pd.read_csv('Dataset/InjuryRecord.csv')
df1 = play_df_cleaned.drop_duplicates('GameID')

problem3_df = injury_df.set_index('GameID').join(df1.set_index(
    'GameID'), how='inner', lsuffix='_left', rsuffix='_right')
y = np.array(problem3_df['BodyPart'])
problem3_df.drop(columns=['PlayKey_left', 'PlayerKey_left', 'PlayerKey_right',
                          'PlayKey_right', 'PlayerGamePlay', 'BodyPart'], inplace=True)
problem3_df1 = pd.get_dummies(problem3_df, dummy_na=False)
X = np.array(problem3_df1)

res = RandomOverSampler(random_state=0, sampling_strategy={
                        'Knee': 480, 'Foot': 70, 'Ankle': 420, 'Heel': 30, 'Toes': 70})
X_resampled, y_resampled = res.fit_resample(X, y)


# Logistic Regression Multiclass Classification

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=21, shuffle=True)
unique, counts = np.unique(y_test, return_counts=True)
print(dict(zip(unique, counts)))
print("Logistic Regression")
new1 = LogisticRegression(max_iter=500)
new1.fit(X_train, y_train)
#y_pred = new1.predict(X_test)
y_pred = new1.predict(X)
#accuracy = accuracy_score(y_test, y_pred)
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)
print('Accuracy: {}'.format(accuracy))
print('Confusion Matrix: \n {}'.format(conf_matrix))
plot_confusion_matrix(new1, X, y)


print("Macro Values")
arr = precision_recall_fscore_support(y, y_pred, average='macro')
print("Precision")
print(arr[0])
print("Recall")
print(arr[1])
print("F Score")
print(arr[2])

print("Micro Values")
arr = precision_recall_fscore_support(y, y_pred, average='micro')
print("Precision")
print(arr[0])
print("Recall")
print(arr[1])
print("F Score")
print(arr[2])


n_classes = 5


y_score = new1.predict_proba(X)
# print(y_score)

y_test = label_binarize(y, classes=['Ankle', 'Foot', 'Heel', 'Knee', 'Toes'])
# Compute ROC curve and ROC area for each class


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw = 1
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['green', 'darkorange', 'cornflowerblue', 'red', 'yellow'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


# Decision Tree  Multiclass Classification
skf = StratifiedKFold(n_splits=2)

for train_index, test_index in skf.split(X_resampled, y_resampled):

    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]

    new = DecisionTreeClassifier(max_depth=8)
    new.fit(X_train, y_train)

    y_pred = new.predict(X)
    accuracy = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)
    print("DecisionTree")
    accuracy = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)
    print('Accuracy: {}'.format(accuracy))
    print('Confusion Matrix: \n {}'.format(conf_matrix))
    plot_confusion_matrix(new1, X, y)

    print("Macro Values")
    arr = precision_recall_fscore_support(y, y_pred, average='macro')
    print("Precision")
    print(arr[0])
    print("Recall")
    print(arr[1])
    print("F Score")
    print(arr[2])

    print("Micro Values")
    arr = precision_recall_fscore_support(y, y_pred, average='micro')
    print("Precision")
    print(arr[0])
    print("Recall")
    print(arr[1])
    print("F Score")
    print(arr[2])

    n_classes = 5

    y_score = new1.predict_proba(X)
    # print(y_score)

    y_test = label_binarize(
        y, classes=['Ankle', 'Foot', 'Heel', 'Knee', 'Toes'])
    # Compute ROC curve and ROC area for each class

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())

    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 1
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['green', 'darkorange', 'cornflowerblue', 'red', 'yellow'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

# MLP for Multiclass Problem


X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=21, shuffle=True)
clf = MLPClassifier(random_state=1).fit(X_train, y_train)
y_pred = clf.predict(X)


accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)
print('Accuracy: {}'.format(accuracy))
print('Confusion Matrix: \n {}'.format(conf_matrix))
plot_confusion_matrix(new1, X, y)


print("Macro Values")
arr = precision_recall_fscore_support(y, y_pred, average='macro')
print("Precision")
print(arr[0])
print("Recall")
print(arr[1])
print("F Score")
print(arr[2])

print("Micro Values")
arr = precision_recall_fscore_support(y, y_pred, average='micro')
print("Precision")
print(arr[0])
print("Recall")
print(arr[1])
print("F Score")
print(arr[2])


n_classes = 5


y_score = new1.predict_proba(X)
# print(y_score)

y_test = label_binarize(y, classes=['Ankle', 'Foot', 'Heel', 'Knee', 'Toes'])
# Compute ROC curve and ROC area for each class


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw = 1
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['green', 'darkorange', 'cornflowerblue', 'red', 'yellow'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()