from data_handler import DataHandler
from models import *


def main():
    handler = DataHandler('data/cardio_train.csv')
    handler.train_test_split()
    # handler.data_exploration()

    model = XGBoostModel(handler.x_train, handler.y_train)
    model.train()

    print(handler.data.info)

    for col in handler.data.columns:
        print(handler.data[col].value_counts())
    print(handler.data.isnull().sum())

    rfc_model = RandomForest(handler)
    rfc_model.prep_forest(10)
    rfc_model.predict()
    rfc_model.hyperparameter_tuning()


if __name__ == "__main__":
    main()
