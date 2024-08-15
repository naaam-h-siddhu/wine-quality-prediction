# importing required modules
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import  SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
sns.set_theme()

#importing dataset
df = pd.read_csv('winequality.csv')

#converting into binary data
bins = [2, 6.5, 8]
tyopes =[ 'bad', 'good']
df['quality'] = pd.cut(df['quality'], bins=bins , labels=tyopes)
label_quality = LabelEncoder()
df['quality'] = label_quality.fit_transform(df['quality'])
df['quality'].value_counts()

x = df.drop('quality', axis=1)
y = df.quality

# splitting data for fitting it in models
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=5, test_size=0.20)

# feature scaling data using standard scaler
sc = StandardScaler()
X_train2 = pd.DataFrame(sc.fit_transform(x_train))
X_test2 = pd.DataFrame(sc.transform(x_test))
X_train2.columns = x_train.columns.values
X_test2.columns = x_test.columns.values
X_train2.index = x_train.index.values
X_test2.index = x_test.index.values
X_train = X_train2
X_test = X_test2

pca = PCA(n_components = 4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# 1 Logistic regression
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



# 2 SVM Linear
clf = SVC(random_state=0, kernel='linear',class_weight='balanced')
clf .fit(X_train,y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results = results._append(model_results, ignore_index = True)



# 3 using SVC with rbf kernal
clf = SVC(random_state = 0, kernel = 'rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results = results._append(model_results, ignore_index = True)


# 4 Randomforest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state = 0, n_estimators = 100,
                                    criterion = 'entropy')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
results = results._append(model_results, ignore_index = True)
print(results)