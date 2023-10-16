from data_handler import DataHandler
from models import XGBoostModel


def main():
    handler = DataHandler('data/cardio_train.csv')
    handler.train_test_split()
    # handler.data_exploration()

    model = XGBoostModel(handler.x_train, handler.y_train)
    model.train()


if __name__ == "__main__":
    main()
