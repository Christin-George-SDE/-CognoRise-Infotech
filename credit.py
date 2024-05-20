# Importing all usefull packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset to a pandas DataFrame
credit_card_data = pd.read_csv('/content/creditcard.csv')

# Printing the fist 5 Rows of data
credit_card_data.head(5)

# Printing the last 5 Rows
credit_card_data.tail(5)

# Credit card information
credit_card_data.info()

# Checking the no of missing values in a column
credit_card_data.isnull().sum()

# Distribution Legit Transaction and Fradulent transaction
 credit_card_data['Class'].value_counts()

"""This dataset is hoghly Unbalanced

0 <- Normal Transaction
1 <- Fraud Transaction
"""

# Seperating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

# Statistical messures of the data
legit.Amount.describe()

fraud.Amount.describe()

# Compare the values for both transactions
credit_card_data.groupby('Class').mean()

"""Under-samlinging

Build a sample dataset containing similar distribution of normal transaction and Fraudulent Transactions

Number of fradulant transaction is 92
"""

legit_sample = legit.sample(n=142)

"""Concatenating two dataframes"""

new_dataset = pd.concat([legit_sample,fraud], axis=0)

new_dataset.head(5)

new_dataset.tail(5)

new_dataset["Class"].value_counts()

new_dataset.groupby('Class').mean()

"""Splitting the data into Features and Traget"""

x = new_dataset.drop(columns='Class', axis=1)
y = new_dataset['Class']

print(x)

print(y)

"""Split the data into triang Data and testing Data"""

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

print(x.shape, x_train.shape, x_test.shape)

"""Model traing

Logistic Regression
"""

model = LogisticRegression()

# Trainig the logistic Regression Model With Training Data
model.fit(x_train, y_train)

"""Model Evaluation

Accuracy Score
"""

# Accuracy on Traiing data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# Accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

print('Accuracy score on Test Data : ', test_data_accuracy)
