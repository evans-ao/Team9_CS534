from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix

from sklearn import metrics 

import pandas as pd 

import matplotlib.pyplot as plt 
import seaborn as sns 


def main():

    filepath = 'data/cardio_train.csv'
    data = pd.read_csv(filepath, delimiter=';')

    diab_cols = ['ap_hi', 'ap_lo', 'cholesterol', 'gluc','smoke','active'] 

    x = data[diab_cols]
    y = data['cardio']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    #Training LR model with hyperparameters
    logreg =  LogisticRegression(solver='liblinear',penalty='l2', C=1.0)
    logreg.fit(x_train,y_train)

    y_pred=logreg.predict(x_test) 

    print('Accuracy: %.2f' % metrics.accuracy_score(y_test, y_pred)) 
    print('precision: %.2f' %  metrics.precision_score(y_test, y_pred))
    print('recall: %.2f' % metrics.recall_score(y_test, y_pred))
    print('f1_score: %.2f' % metrics.f1_score(y_test, y_pred))

    #Plotting the confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    

if __name__ == "__main__":
    main()
