import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import category_encoders as ce
from data_handler import DataHandler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RandomForest:
    def __init__(self,data_handler):
        self.handler = data_handler
        self.rfc = None

    def print_train_data(self):
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

    def print_feature_importance(self):
        feature_scores = pd.Series(
            self.rfc.feature_importances_, index=self.handler.x_train.columns).sort_values(ascending=False)
        print(feature_scores)


def main():
    handler = DataHandler('data/cardio_train.csv')
    handler.train_test_split()
    handler.data_exploration()

    # model = XGBoostModel(handler.data.drop(columns=['cardio','id']), handler.data['cardio'])
    # model.train()
    # model.cross_validate(k=10)
    # model.hyperparameter_tuning()

    # for col in handler.data.columns:
    #    print(handler.data[col].value_counts())

    # print(handler.data.isnull().sum())

    rfc_model = RandomForest(handler)
    rfc_model.prep_forest(10)
    rfc_model.predict()
    rfc_model.print_feature_importance()


if __name__ == "__main__":
    main()