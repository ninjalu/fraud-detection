# fraud-detection
  
## Task and data
### Data
* The data consists of genuine and fraudulent transactions over 2 days. <br>
* Highly imbalanced: only 0.172% of the transactions are fraudulent.<br>
* There are 28 anonymous features in the dataset, all standardised with a mean of 0 and standard deviation of 1. <br>
* Additional features include Time (seconds since the first transaction) and Amount (transaction amount). <br>

### Task and challenge
* Build a ML model to identify fraudulent transactions from normal transactions <br>
* The imbalanced data is challenging in terms of model training. Traditional machine learning techniques do not like extremely imbalanced data. This could potentially be resolved using over sampling and under sampling. <br>
* Scoring metric for training and evaluating needs to be carefully considered. (Predicting all data points as normal will give you a 99.82% accuracy) <br>

## Preprocessing steps
* I conclude the first look with the following preprocessing steps: <br>
1. Normalise amount
2. Transform time into hours from hour 0
3. We have later found out that hours/time is not predictive in fraud. Perhaps there happens to be peaks just after hour 0 in the 2 days sampled. That same pattern didn’t show in the test set.
* Note outliers: because this is a prediction task, not an analysis task, I decided not to delete the outliers because I want to build a model that can deal with outliers, particularly if the outliers are correlated with fraud cases.

## Model choices
Because this is a classification problem, I short listed these models to try: <br>
1. Basic logistic regression
2. KNN
3. Random forest
4. SVC
5. XGBoost

## Sampling choices
* For resolving issues that could arise from imbalance, I tried combinations of these over and under sampling methods: <br>
1. SMOTE: Synthetic Minority Oversampling Technique
2. BorderlineSMOTE: SMOTE that samples more around the class borderline
3. Randomised under sampling
* I chose NOT to resample the validation set, because I want to make sure the validation set is as close to the test set as possible.
* With the limited time I got to play around with resampling, they actually produced worse results, compared to no resampling, particularly under sampling completely collapsed precision score on validation.

## Metrics
* Because I was working with severely imbalanced data, I choose the metrics that only deal with precision (FP) and recall (FN), fraud being positive. <br>
* The area under precision and recall curve is used.<br>
* Also I devised a metric that takes into account the cost of FP (client annoyed at the hassle, losing business), and FN (fraudulent transaction not detected, losing money)<br>

## Random search CV
* For KNN, random forest, SVM and XGBoost, I used Sklearn’s RandomSearchCV to cross validate and tune hyper parameters.<br>
* XGBoost produced the best result, 1.00 train, 0.84 test. <br>
* Questions on how to narrow the train and validation loss gap, and how to improve random search.<br>

## What’s next
1. The customised cost based metric is quite interesting, and it would be good to work out an interpretation of the score in money.
2. Experiment on grid search and see if it improves the train/validation gap
3. Experiment on early stopping
4. Experiment on NN for fraud detection
5. Dive deep into the meaning of quasi-separation; why is it; and how to deal with it.
6. Experiment with less features.
