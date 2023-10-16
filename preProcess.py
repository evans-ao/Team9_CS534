import pandas as pd


def dataExploration(data):
    rows, column = data.shape
    print('Number of rows: ', rows)
    print('Number of rows: ', column)

    print('Dataframe head')
    print(data.head())

    missing_values_per_column = data.isnull().sum()
    print('Missing values')
    print(missing_values_per_column)

    cardio_counts = data['cardio'].value_counts()
    print(cardio_counts)


def load_data(filepath):
    return pd.read_csv('cardio_train.csv', delimiter=';')


def main():
    data = load_data('cardio_train.csv')
    # dataExploration(data)


if __name__ == "__main__":
    main()
