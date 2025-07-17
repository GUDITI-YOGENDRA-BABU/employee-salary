#employee salary predication using adult csv
#load your libarary
import pandas as pd
data=pd.read_csv(r"C:\Users\gudit\Downloads\RECOVER\adult 3.csv")
print(data.occupation.value_counts())
data.occupation=data.occupation.replace({'?':'others'})
data.workclass=data.workclass.replace({'?':'NOT LISTED'})
data['native-country']=data['native-country'].replace({'?':'NOT COUNTABLE'})
data=data[data['workclass']!='Without-pay']
data=data[data['workclass']!='Never-worked']
data=data[data['occupation']!='Armed-Forces']
data=data[data['occupation']!='Priv-house-serv']
data=data[data['education']!='7th-8th']
data=data[data['education']!='Preschool']
data=data[data['education']!='1st-4th']
data=data[data['education']!='5th-6th']
data=data[data['education']!='9th']
data=data[data['native-country']!='Holand-Netherlands']
data=data[data['native-country']!='Hungary']
data=data[data['native-country']!='Honduras']
data=data[data['native-country']!='Laos']
data=data[data['marital-status']!='Scotland']
data=data[data['marital-status']!='Yugoslavia']
print(data.occupation.value_counts())
print(data.workclass.value_counts())
print(data.education.value_counts())
print(data.race.value_counts())
print(data.gender.value_counts())
print(data['marital-status'].value_counts())
print(data['native-country'].value_counts())
data.drop(columns=['education'],inplace=True)
data=data[(data['age']<=70)&(data['age']>=20)]
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['workclass']=encoder.fit_transform(data['workclass'])
data['marital-status']=encoder.fit_transform(data['marital-status'])
data['occupation']=encoder.fit_transform(data['occupation'])
data['relationship']=encoder.fit_transform(data['relationship'])
data['race']=encoder.fit_transform(data['race'])
data['gender']=encoder.fit_transform(data['gender'])
data['native-country']=encoder.fit_transform(data['native-country'])
data=data[(data['workclass']<=5)&(data['workclass']>=1)]
data=data[(data['marital-status']<=5)&(data['marital-status']>=1)]
data=data[(data['native-country']<=35)&(data['native-country']>=5)]
x=data.drop(columns=['income'])
y=data['income']
print(y)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x)
print(x)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=30,stratify=y)
print(xtrain)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(xtrain,ytrain)#input and output training data
predict=knn.predict(xtest)
print(predict)
from sklearn.metrics import accuracy_score
accuracy_score(ytest,predict)
print(accuracy_score)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)#input and output training data
predict1=lr.predict(xtest)
print(predict1)
from sklearn.metrics import accuracy_score
acc1=accuracy_score(ytest,predict1)
print(acc1)
from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(solver='adam',hidden_layer_sizes=(5,2),random_state=2,max_iter=2000)
clf.fit(xtrain,ytrain)
predict2=clf.predict(xtest)
print(predict2)
from sklearn.metrics import accuracy_score
acc2=accuracy_score(ytest,predict2)
print(acc2)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler,OneHotEncoder

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=35)

models={
    "LogisticRegression":LogisticRegression(),
    "RandomForest":RandomForestClassifier(),
    "KNN":KNeighborsClassifier(),
    "SVM":SVC(),
    "GradientBoosting":GradientBoostingClassifier()
}
results={}
for name,model in models.items():
    pipe=Pipeline([
        ('scaler',StandardScaler()),
        ('model',model)
    ])

    pipe.fit(xtrain,ytrain)
    y_pred=pipe.predict(xtest)
    acc3=accuracy_score(ytest,y_pred)
    results[name]=acc3
    print(f"{name} Accuracy:{acc3:.4f}")
    print(classification_report(ytest,y_pred))

import matplotlib.pyplot as plt
plt.bar(results.keys(),results.values(),color='skyblue')
plt.ylabel('Accuracy Score')
plt.title('Model comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Train-test split
xtrain,xtest,ytrain,ytest= train_test_split(x, y, test_size=0.2, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(xtrain,ytrain)
    preds = model.predict(xtest)
    acc = accuracy_score(ytest, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Get best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n✅ Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save the best model
joblib.dump(best_model, "best_model.pkl")
print("✅ Saved best model as best_model.pkl")