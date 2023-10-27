import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets using the split_data function
def split_data(input_file, train_output_file, test_output_file, test_size=0.2, random_seed=None):
    # Load CSV data into a pandas DataFrame with semicolon (;) as the delimiter
    data = pd.read_csv(input_file, delimiter=';')

    # Check for and handle missing or extra rows
    if len(data) < 2:
        raise ValueError("Insufficient data for splitting. At least 2 rows are required.")

    # Features (X) exclude the 'cardio' column, which is the label we want to predict
    features = data.drop(columns=['cardio'])
    # Labels (y) are the 'cardio' column
    labels = data['cardio']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_seed)

    # Check if lengths of training and testing datasets are the same
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Mismatched lengths of training and testing datasets.")

    # Save training and testing data to new CSV files
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data.to_csv(train_output_file, index=False, sep=';')  # Save with semicolon (;) as delimiter
    test_data.to_csv(test_output_file, index=False, sep=';')  # Save with semicolon (;) as delimiter

# Split data and save into train.csv and test.csv
split_data('cardio_train.csv', 'train.csv', 'test.csv', test_size=0.2, random_seed=42)

# Load training data from train.csv using semicolon (;) as delimiter
train_data = pd.read_csv('train.csv', delimiter=';')

# Features: age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
# Target: cardio
features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
target = 'cardio'

# Split the data into features (X) and target (y)
X_train = train_data[features]
y_train = train_data[target]

# Load test data from test.csv using semicolon (;) as delimiter
test_data = pd.read_csv('test.csv', delimiter=';')

# Prepare test features and labels
X_test = test_data[features]
y_test = test_data[target]
for i in range(1, 101):
    # Initialize the KNN classifier with k=3
    knn_classifier = KNeighborsClassifier(n_neighbors=i)

    # Train the KNN classifier
    knn_classifier.fit(X_train, y_train)

    # Use the trained classifier to make predictions
    predictions = knn_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f'{accuracy * 100:.2f}%')