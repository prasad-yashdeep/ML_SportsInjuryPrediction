

# Sports Injury Prediction using Machine Learning Techniques

Predicting injury beforehand would be a huge help to the players, ultimately revolutionising the sports industry.Knowing the resting period in advance, would even help teams strategize in a better manner for future tournaments.We also aim to tell the players in advance about the body part more likely to be injured so that the players can take prior measures in order to prevent those injuries.

## Data Pre Processing Techniques
1) Non-numeric data of certain features were converted to binary data for effective model training.
2) We joined injury data with play data using different joining techniques to increase the number of attributes in our final dataset.
3) We oversampled the data in order to increase the instances of injuries.This will help us to find the number of days of rest and the body part more likely to get injured in an occurrence of injury.
4) We replaced the missing values of temperature with the average temperature value. 
5) Our objective was to detect whether the player has suffered an injury or not however there was no field in the original dataset for the following , so we made use of the attributes DM_M1, DM_M7,DM_28,DM_42 , and if any one of them had a non-zero value , henceforth an injury has occured and therefore it is an instance of injury.
6) Some play days were negative due to some erroneous data so their absolute values were taken as play days can never be negative.


## Problem 1
In this problem we perfromed binary classification to predict whether the player suffered from injury or not based on uncorrelated attributes like Stadium Type,Field Type,Weather,etc.We used `Gaussian Naive Bayes,Logistic Regression,Decision Tree and XGBoost` for this task.To implement this we run problem1.py using the below command.

## Problem 2
In this problem we performed Regression to find out the number of days of rest needed by the player in occurence of an injury.We performed `Linear Regression,Support Vector Machine and Random Forest Regressor` to perform this task.To implement this we run problem2.py using the below command.

## Problem 3
In this problem,we predict the body part most likely to be injured based on other attributes.We performed `Logistic Regression,Decision Tree and Multi Layer Perceptron` to perform this task.To implement this we run problem3.py using the below command.





# How To run the Project-:

Run all the files using the command-:
### `python3 problem1.py`
### `python3 problem2.py`
### `python3 problem3.py`




