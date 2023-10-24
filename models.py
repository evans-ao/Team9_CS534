from sklearn.model_selection import cross_val_score, GridSearchCV
from tqdm import tqdm
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import sort
from sklearn.feature_selection import SelectFromModel
import warnings


class XGBoostModel:
    # https://www.kaggle.com/code/gauthamyaramasa/cardiovascular-disease-prediction-xgboost/notebook
    # commenting the code which uses train test splits. Doing 10-fold cross validation
    """
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

        plot_importance(model)
        plt.show()
        # id is the most important feature according to the plot. Removed id from the dataframe.

        # Fit model using each importance as a threshold
        thresholds = sort(model.feature_importances_)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for thresh in thresholds:
                # select features using threshold
                selection = SelectFromModel(model, threshold=thresh, prefit=True)
                select_X_train = selection.transform(self.x_train)
                # train model
                selection_model = XGBClassifier()
                selection_model.fit(select_X_train, self.y_train)
                # eval model
                select_X_test = selection.transform(self.x_test)
                y_pred = selection_model.predict(select_X_test)
                predictions = [round(value) for value in y_pred]
                accuracy = accuracy_score(self.y_test, predictions)
                print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))
        """

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    """def select_important_features(self, threshold=0.01):
        # Train the model on the entire dataset
        model = XGBClassifier()
        model.fit(self.x_data, self.y_data)

        # Select features based on threshold
        selection = SelectFromModel(model, threshold=threshold, prefit=True)
        # plot_importance(model)
        # plt.show()

        # Printing feature importance scores
        feature_importance = model.feature_importances_
        sorted_idx = feature_importance.argsort()[::-1]  # Sort in descending order
        for index in sorted_idx:
            print(f"Feature: {self.x_data.columns[index]}, Importance: {feature_importance[index]}")
        return selection.transform(self.x_data)

    def cross_validate(self, k=10, feature_threshold=0.03):
        model = XGBClassifier()
        # to perform feature selection, we need to train the model on entire data.
        reduced_x_data = self.select_important_features(threshold=feature_threshold)
        scores = cross_val_score(model, reduced_x_data, self.y_data, cv=k, scoring='accuracy', n_jobs=-1)

        for i, score in enumerate(tqdm(scores, desc="Cross-validation", unit="fold")):
            print(f"Fold {i + 1} Accuracy: {score:.4f}")

        print(f"Average Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")"""

    def hyperparameter_tuning(self, k=10):
        param_grid = {
            'max_depth': [3, 4, 5],
            'subsample': [0.6, 0.8, 1.0],
            'learning_rate': [0.01, 0.02, 0.05, 0.1]
        }

        model = XGBClassifier()

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                   scoring='accuracy', cv=k, verbose=1, n_jobs=-1)

        grid_search.fit(self.x_data, self.y_data)

        print(f"Best accuracy: {grid_search.best_score_} using {grid_search.best_params_}")

        return grid_search.best_estimator_
