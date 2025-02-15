import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time as t
import sklearn.utils as u
import sklearn.preprocessing as pp
import sklearn.tree as tr
import sklearn.ensemble as es
import sklearn.metrics as m
import sklearn.linear_model as lm
import sklearn.neural_network as nn
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings as w
w.filterwarnings('ignore')

data = pd.read_csv("AI-Data.csv")
ch = 0
while(ch != 10):
    print("1.Marks Class Count Graph\t2.Marks Class Semester-wise Graph\n3.Marks Class Gender-wise Graph\t4.Marks Class Nationality-wise Graph\n5.Marks Class Grade-wise Graph\t6.Marks Class Section-wise Graph\n7.Marks Class Topic-wise Graph\t8.Marks Class Stage-wise Graph\n9.Marks Class Absent Days-wise\t10.No Graph\n")
    ch = int(input("Enter Choice: "))
    if (ch == 1):
        print("Loading Graph....\n")
        t.sleep(1)
        print("\tMarks Class Count Graph")
        axes = sb.countplot(x='Class', data=data, order=['L', 'M', 'H'])
        plt.show()
    elif (ch == 2):
        print("Loading Graph....\n")
        t.sleep(1)
        print("\tMarks Class Semester-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='Semester', hue='Class', data=data, hue_order=['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch == 3):
        print("Loading Graph..\n")
        t.sleep(1)
        print("\tMarks Class Gender-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='gender', hue='Class', data=data, order=['M', 'F'], hue_order=['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch == 4):
        print("Loading Graph..\n")
        t.sleep(1)
        print("\tMarks Class Nationality-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch == 5):
        print("Loading Graph: \n")
        t.sleep(1)
        print("\tMarks Class Grade-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='GradeID', hue='Class', data=data, order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch ==6):
        print("Loading Graph..\n")
        t.sleep(1)
        print("\tMarks Class Section-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='SectionID', hue='Class', data=data, hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch == 7):
        print("Loading Graph..\n")
        t.sleep(1)
        print("\tMarks Class Topic-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='Topic', hue='Class', data=data, hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch == 8):
        print("Loading Graph..\n")
        t.sleep(1)
        print("\tMarks Class Stage-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='StageID', hue='Class', data=data, hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()
    elif (ch == 9):
        print("Loading Graph..\n")
        t.sleep(1)
        print("\tMarks Class Absent Days-wise Graph")
        fig, axesarr = plt.subplots(1, figsize=(10, 6))
        sb.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order = ['L', 'M', 'H'], axes=axesarr)
        plt.show()
if(ch == 10):
    print("Exiting..\n")
    t.sleep(1)

data_copy = data.copy()

for column in data.columns:
    if data[column].dtype == type(object):
        le = pp.LabelEncoder()
        data[column] = le.fit_transform(data[column])

X = data.drop("Class", axis=1)
y = data["Class"]

smote = SMOTE(sampling_strategy={0: 1000, 1: 2000, 2: 1500}, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_names = []
trained_models = {}

dt_params = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
modelD = GridSearchCV(tr.DecisionTreeClassifier(random_state=42), dt_params, cv=5)
modelD.fit(X_train, y_train)
trained_models['dt'] = modelD.best_estimator_
model_names.append(('dt', modelD.best_estimator_))
lbls_predD = modelD.predict(X_test)
print("\nBest parameters for Decision Tree:", modelD.best_params_)
print("Accuracy measures using Decision Tree:")
print(m.classification_report(y_test, lbls_predD))
print("Accuracy using Decision Tree: ", round(m.accuracy_score(y_test, lbls_predD), 3))
t.sleep(1)

rf_params = {
    'n_estimators': [100, 250, 5000],
    'max_depth': [None, 30, 50, 100],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'min_samples_leaf': [1, 2, 5]
}
modelR = GridSearchCV(es.RandomForestClassifier(random_state=42), rf_params, cv=5)
modelR.fit(X_train, y_train)
trained_models['rf'] = modelR.best_estimator_
model_names.append(('rf', modelR.best_estimator_))
lbls_predR = modelR.predict(X_test)
print("\nBest parameters for Random Forest:", modelR.best_params_)
print("Accuracy Measures for Random Forest Classifier:")
print(m.classification_report(y_test, lbls_predR))
print("Accuracy using Random Forest: ", round(m.accuracy_score(y_test, lbls_predR), 3))
t.sleep(1)

perceptron_params = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'alpha': [0.0001, 0.001, 0.01]
}
modelP = GridSearchCV(lm.Perceptron(random_state=42), perceptron_params, cv=5)
modelP.fit(X_train, y_train)
trained_models['perceptron'] = modelP.best_estimator_
model_names.append(('perceptron', modelP.best_estimator_))
lbls_predP = modelP.predict(X_test)
print("\nBest parameters for Perceptron:", modelP.best_params_)
print("Accuracy measures using Linear Model Perceptron:")
print(m.classification_report(y_test, lbls_predP))
print("Accuracy using Linear Model Perceptron: ", round(m.accuracy_score(y_test, lbls_predP), 3))
t.sleep(1)

lr_params = {
    'C': [0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 200, 500]
}
modelL = GridSearchCV(lm.LogisticRegression(random_state=42), lr_params, cv=5)
modelL.fit(X_train, y_train)
trained_models['lr'] = modelL.best_estimator_
model_names.append(('lr', modelL.best_estimator_))
lbls_predL = modelL.predict(X_test)
print("\nBest parameters for Logistic Regression:", modelL.best_params_)
print("Accuracy measures using Linear Model Logistic Regression:")
print(m.classification_report(y_test, lbls_predL))
print("Accuracy using Linear Model Logistic Regression: ", round(m.accuracy_score(y_test, lbls_predL), 3))
t.sleep(1)

mlp_params = {
    'hidden_layer_sizes': [(100, 50), (200, 100)],
    'activation': ['relu'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.005, 0.001],
}

modelN = GridSearchCV(nn.MLPClassifier(random_state=42, max_iter=1000, learning_rate_init=0.001), mlp_params, cv=5)
modelN.fit(X_train, y_train)
trained_models['mlp'] = modelN.best_estimator_
model_names.append(('mlp', modelN.best_estimator_))
lbls_predN = modelN.predict(X_test)
print("\nBest parameters for MLP Classifier:", modelN.best_params_)
print("Accuracy measures using MLP Classifier:")
print(m.classification_report(y_test, lbls_predN))
print("Accuracy using Neural Network MLP Classifier: ", round(m.accuracy_score(y_test, lbls_predN), 3))
t.sleep(1)

voting_clf = VotingClassifier(estimators=model_names, voting='soft')
voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)
print("\nVoting Classifier Accuracy:")
print(m.classification_report(y_test, voting_pred))
print("Accuracy using Voting Classifier: ", round(m.accuracy_score(y_test, voting_pred), 3))

choice = input("Do you want to test specific input (y or n): ")
if(choice.lower()=="y"):
    gen = input("Enter Gender (M or F): ")
    if (gen.upper() == "M"):
       gen = 1
    elif (gen.upper() == "F"):
       gen = 0
    nat = input("Enter Nationality: ")
    pob = input("Place of Birth: ")
    gra = input("Grade ID as (G-<grade>): ")
    if(gra == "G-02"):
        gra = 2
    elif (gra == "G-04"):
        gra = 4
    elif (gra == "G-05"):
        gra = 5
    elif (gra == "G-06"):
        gra = 6
    elif (gra == "G-07"):
        gra = 7
    elif (gra == "G-08"):
        gra = 8
    elif (gra == "G-09"):
        gra = 9
    elif (gra == "G-10"):
        gra = 10
    elif (gra == "G-11"):
        gra = 11
    elif (gra == "G-12"):
        gra = 12
    sec = input("Enter Section: ")
    top = input("Enter Topic: ")
    sem = input("Enter Semester (F or S): ")
    if (sem.upper() == "F"):
       sem = 0
    elif (sem.upper() == "S"):
       sem = 1
    rel = input("Enter Relation (Father or Mum): ")
    if (rel == "Father"):
       rel = 0
    elif (rel == "Mum"):
       rel = 1
    rai = int(input("Enter raised hands: "))
    res = int(input("Enter Visited Resources: "))
    ann = int(input("Enter announcements viewed: "))
    dis = int(input("Enter no. of Discussions: "))
    sur = input("Enter Parent Answered Survey (Y or N): ")
    if (sur.upper() == "Y"):
       sur = 1
    elif (sur.upper() == "N"):
       sur = 0
    sat = input("Enter Parent School Satisfaction (Good or Bad): ")
    if (sat == "Good"):
       sat = 1
    elif (sat == "Bad"):
       sat = 0
    absc = input("Enter No. of Abscenes(Under-7 or Above-7): ")
    if (absc == "Under-7"):
       absc = 1
    elif (absc == "Above-7"):
       absc = 0
    
    input_data = {
        'gender': gen,
        'NationalITy': 0,  
        'PlaceofBirth': 0,  
        'StageID': 0,
        'GradeID': gra,
        'SectionID': 0,  
        'Topic': 0,  
        'Semester': sem,
        'Relation': rel,
        'raisedhands': rai,
        'VisITedResources': res,
        'AnnouncementsView': ann,
        'Discussion': dis,
        'ParentAnsweringSurvey': sur,
        'ParentschoolSatisfaction': sat,
        'StudentAbsenceDays': absc,
    }
    
    input_df = pd.DataFrame([input_data])
    
    input_df = input_df[X.columns]

