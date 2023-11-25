from Team9_CS534.random_forest_model import RandomForest
from data_handler import DataHandler
from models import XGBoostModel


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
