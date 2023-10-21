from data_handler import DataHandler
from models import XGBoostModel


def main():
    handler = DataHandler('data/cardio_train.csv')
    handler.train_test_split()
    # handler.data_exploration()

    model = XGBoostModel(handler.x_train, handler.y_train, handler.x_test, handler.y_test)
    # Need to work on hyper parameter tuning
    model.train()
    # model.cross_validate(k=10)


if __name__ == "__main__":
    main()
