# Python-DBM (svm-dbm)
Decision Boundary Making (DBM) using Support Vector Machine (SVM) implemented in python3, scikitlearn   
DBM is a machine learning model that has compactness and high performance  
In this script, SVM is used as high performance model of DBM (see Reference)  
DBM is for binary classificaiton probelm (see Reference)  


## How to install 
```
pip install git+https://github.com/masaponto/python-dbm 
```

## Require
- scikit-learn: 0.18.1
- scipy: 0.18.1
- numpy: 1.11.2


## Usage
```python
from svm_dbm import DBM
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing, model_selection
from sklearn.datasets import fetch_mldata

db_name = 'australian'

data_set = fetch_mldata(db_name)
data_set.data = preprocessing.scale(data_set.data)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
        data_set.data, data_set.target, test_size=0.4)

mlp = MLPClassifier(solver='sgd', alpha=1e-5,
        hidden_layer_sizes=(2,), activation='logistic', learning_rate_init=0.5)

dbm = DBM(mlp).fit(X_train, y_train)
print("DBM-MLP Accuracy %0.3f " % dbm.score(X_test, y_test))
```


## Reference
https://www.jstage.jst.go.jp/article/ipsjjip/23/4/23_497/_article  
