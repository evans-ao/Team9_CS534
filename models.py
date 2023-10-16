class XGBoostModel:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def train(self):
        print(self.y_train)