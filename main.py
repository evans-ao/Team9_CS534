from data_handler import DataHandler
from models import XGBoostModel


def main():
    handler = DataHandler('data/cardio_train.csv')
    # handler.train_test_split()
    # handler.data_exploration()

    model = XGBoostModel(handler.data.drop(columns=['cardio','id']), handler.data['cardio'])
    # model.train()
    # model.cross_validate(k=10)
    model.hyperparameter_tuning()


if __name__ == "__main__":
    main()
