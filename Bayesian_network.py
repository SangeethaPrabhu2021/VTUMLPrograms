"""8. Write a program to construct a Bayesian network considering medical data. Use this model to
demonstrate the diagnosis of heart patients using standard Heart Disease Data Set."""
import numpy as np
import pandas as pd
import csv 
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

#heartDisease = pd.read_csv('8heartdiseasedata.csv')

data=pd.read_csv("heartdisease.csv", names=['age','Gender','Family','diet','Lifestyle','cholestrol','heartdisease'])
# heartDisease = heartDisease.replace('?',np.nan)
heartDisease=pd.DataFrame(data)

from sklearn.preprocessing import LabelEncoder

heartDisease.columns
 
lb = LabelEncoder()
for col in heartDisease.columns:
    heartDisease[col] = lb.fit_transform(heartDisease[col])



print('Sample instances from the dataset are given below')
print(heartDisease.head())

print('\n Attributes and datatypes')
print(heartDisease.dtypes)

model= BayesianModel([('age','heartdisease'),('Gender','heartdisease'),('Family','heartdisease'),('diet','cholestrol'),('Lifestyle','diet'),('heartdisease','cholestrol')])
print('\nLearning CPD using Maximum likelihood estimators')
model.fit(heartDisease,estimator=MaximumLikelihoodEstimator)

print('\n Inferencing with Bayesian Network:')
HeartDiseasetest_infer = VariableElimination(model)


print('For age Enter { SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4 }')
print('For Gender Enter { Male:0, Female:1 }')
print('For Family History Enter { yes:1, No:0 }')
print('For diet Enter { High:0, Medium:1 }')
print('For lifeStyle Enter { Athlete:0, Active:1, Moderate:2, Sedentary:3 }')
print('For cholesterol Enter { High:0, BorderLine:1, Normal:2 }')

print('\n 1. Probability of HeartDisease given evidence= age')
q1=HeartDiseasetest_infer.query(variables=['heartdisease'],evidence={'age':int(input("Enter age"))})
print(q1)

print('\n 2. Probability of HeartDisease given evidence= cholestrol ')
q2=HeartDiseasetest_infer.query(variables=['heartdisease'],evidence={'cholestrol':int(input("Enter Cholestrol"))})
print(q2)
