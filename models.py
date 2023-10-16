import xgboost as xgb
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


class XGBoostModel:
    # https://www.kaggle.com/code/gauthamyaramasa/cardiovascular-disease-prediction-xgboost/notebook
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train(self):
        model = XGBClassifier()
        model.fit(self.x_train, self.y_train)
        predict = model.predict(self.x_test)
        # Create the confusion matrix
        cm = confusion_matrix(self.y_test, predict)

        # Plot the confusion matrix heatmap
        plt.figure(figsize=(6, 4))
        fg = sns.heatmap(cm, annot=True, cmap="Reds", fmt='d')
        figure = fg.get_figure()

        # Add labels and title
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title("Output Confusion Matrix")
        plt.show()

        print('True Positive Cases : {}'.format(cm[1][1]))
        print('True Negative Cases : {}'.format(cm[0][0]))
        print('False Positive Cases : {}'.format(cm[0][1]))
        print('False Negative Cases : {}'.format(cm[1][0]))

        pre = cm[1][1] / (cm[1][1] + cm[0][1])
        rec = cm[1][1] / (cm[1][1] + cm[1][0])
        f1_score = 2 * (pre * rec) / (pre + rec)

        print("The Precision is:", round(pre, 3))
        print("The Recall is:", round(rec, 3))
        print("The F1 Score is:", round(f1_score, 3))

        print("The Model Accuracy is:", round(accuracy_score(self.y_test, predict) * 100, 3), "%")

    def cross_validate(self, k=5):
        model = XGBClassifier()
        scores = cross_val_score(model, self.x_train, self.y_train, cv=k, scoring='accuracy', n_jobs=-1)

        for i, score in enumerate(tqdm(scores, desc="Cross-validation", unit="fold")):
            print(f"Fold {i + 1} Accuracy: {score:.4f}")

        print(f"Average Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
