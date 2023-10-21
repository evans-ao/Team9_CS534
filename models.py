import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # statistical data visualization %matplotlib inline
import category_encoders as ce
from data_handler import DataHandler

# import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



class XGBoostModel:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def train(self):
        print(self.y_train)
        print("__________________")
        print(self.x_train)


class RandomForest:
    def __init__(self,data_handler):
        self.handler = data_handler
        self.rfc = None

    def train(self):
        print(self.handler.y_train)
        print("__________________")
        print(self.handler.x_train)

    def prep_forest(self, num_trees):
        self.rfc = RandomForestClassifier(n_estimators=num_trees, random_state=0)
        encoder = ce.OrdinalEncoder(self.handler.data.columns)
        self.handler.x_train = encoder.fit_transform(self.handler.x_train)
        # print(self.handler.x_train)
        self.rfc.fit(self.handler.x_train, self.handler.y_train)

    def predict(self):
        y_predict = self.rfc.predict(self.handler.x_test)
        print('Model accuracy score with ' + str(self.rfc.n_estimators) +
              ' decision-trees : {0:0.4f}'.format(accuracy_score(self.handler.y_test, y_predict)))

    def hyperparameter_tuning(self):
        feature_scores = pd.Series(
            self.rfc.feature_importances_, index=self.handler.x_train.columns).sort_values(ascending=False)
        print(feature_scores)


