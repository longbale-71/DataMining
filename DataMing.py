

import pandas as pd
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as pl

wine= pd.read_csv("./diabetes_data_upload.csv")
Gender={'Male':0,'Female':1}
wine['Gender']=wine['Gender'].map(Gender)


Polyuria={'Yes':0,'No':1}
wine['Polyuria']=wine['Polyuria'].map(Polyuria)

Polydipsia={'Yes':0,'No':1}
wine['Polydipsia']=wine['Polydipsia'].map(Polydipsia)

SuddenWeightLoss={'Yes':0,'No':1}
wine['SuddenWeightLoss']=wine['SuddenWeightLoss'].map(SuddenWeightLoss)

weakness={'Yes':0,'No':1}
wine['weakness']=wine['weakness'].map(weakness)

Polyphagia={'Yes':0,'No':1}
wine['Polyphagia']=wine['Polyphagia'].map(Polyphagia)

GenitalThrush={'Yes':0,'No':1}
wine['GenitalThrush']=wine['GenitalThrush'].map(GenitalThrush)

visualBlurring={'Yes':0,'No':1}
wine['visualBlurring']=wine['visualBlurring'].map(visualBlurring)

Itching={'Yes':0,'No':1}
wine['Itching']=wine['Itching'].map(Itching)

Irritability={'Yes':0,'No':1}
wine['Irritability']=wine['Irritability'].map(Irritability)

delayedHealing={'Yes':0,'No':1}
wine['delayedHealing']=wine['delayedHealing'].map(delayedHealing)

partialParesis={'Yes':0,'No':1}
wine['partialParesis']=wine['partialParesis'].map(partialParesis)

muscleStiffness={'Yes':0,'No':1}
wine['muscleStiffness']=wine['muscleStiffness'].map(muscleStiffness)

Alopecia={'Yes':0,'No':1}
wine['Alopecia']=wine['Alopecia'].map(Alopecia)

Obesity={'Yes':0,'No':1}
wine['Obesity']=wine['Obesity'].map(Obesity)

result={'Positive':0,'Negative':1}
wine['result']=wine['result'].map(result)

data =['Age','Gender','Polyuria','Polydipsia','SuddenWeightLoss','weakness',
        'Polyphagia','GenitalThrush','visualBlurring','Itching','Irritability','delayedHealing',
        'partialParesis','muscleStiffness','Alopecia','Obesity']

x=wine[data]
y=wine['result']

model= tree.DecisionTreeClassifier()
model=model.fit(x,y)
fig=pl.figure(figsize=(250,200))
_=tree.plot_tree(model)





