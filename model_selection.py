"""
This module uses the following models to find the best performing model and parameters:
1. KNN with borderline SMOTE and random undersampling
2. SVM with alternative borderline SMOTE and random undersampling
3. RF with borderline SMOTE and random undersampling
4. XGB with borderline SMOTE and random undersampling
5. Logistic regression with borderline SMOTE and random undersampling

How to use the class EnsemModel:
run .sample(X, y) -- sampling the data as per above
run .select_metric() -- select a metrics for evaluation
run .fit() -- performs hyper parameter tuning with the training data and RandomSearchCV
run .fit_predict_best_model -- fit and predict with the best performing model on all training data

    """

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier
import preprocessing


class EnsemModel:
    ''' 
    This class performs hyperparameter tuning for each model
    Evaluates model performances
    And saves best performers
    '''

    def __init__(self):
        '''
        Initiate training and validation sets
        Library containing hyperparameters for all models
        Library containing evaluation metrics/scores
        '''

        self.hyperP_best = {
            'KNN': None,
            'SVC': None,
            'RF': None,
            'XGB': None
        }

        self.hyperP_search = {
            'KNN': {
                'n_neighbors': range(1, 21, 2),
                'metric': ['euclidean', 'manhattan']
            },
            'SVC': {
                'C': [100, 10, 1.0, 0.1, 0.001]
            },
            'RF': {
                'n_estimators': [100, 300, 500, 800, 1200],
                'max_depth': [5, 8, 15, 25, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5, 10],
                'bootstrap': [True, False]
            },
            'XGB': {
                'n_estimators': [100, 500, 1000],
                'max_depth': [1, 5, 10],
                'min_child_weight': [1, 5, 10],
                'subsample': [0.8, 1],
                'colsample_bytree': [0.8, 1],
                'eta': [0.001, 0.01, 0.1, 0.2],
                'gamma': [0.5, 1, 2, 5]
            },
            'LG': {
                'penalty': ['l1', 'l2']
            }
        }

        self.score = {
            'KNN': None,
            'SVC': None,
            'RF': None,
            'XGB': None
        }

        self.best_model = None
        self.predictions = None
        self.test_score = None

        self.X = None
        self.y = None
        self.X_svc = None
        self.y_svc = None
        self.metric = None

    def resample(self, X, y):
        self.X, self.y = preprocessing.sampler(X, y)
        self.X_svc, self.y_svc = preprocessing.svmsampler(X, y)

    def select_metrics(self, func):
        self.metric = make_scorer(func)

    def fit(self):
        '''
        Fit X y train and validation to each ensemble model
        Save the best hyperparameters and evaluation metrics
        '''
        knn = KNeighborsClassifier()
        knn_search = RandomizedSearchCV(
            estimator=knn,
            param_distributions=self.hyperP_search['KNN'],
            scoring=self.metric,
            n_iter=500,
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1)
        knn_search.fit(self.X, self.y)
        self.hyperP_best['KNN'] = knn_search.best_params_
        self.score['KNN'] = knn_search.best_score_

        svc = SVC()
        svc_search = RandomizedSearchCV(
            estimator=svc,
            param_distributions=self.hyperP_search['SVC'],
            scoring=self.metric,
            n_iter=500,
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1)
        svc_search.fit(self.X_svc, self.y_svc)
        self.hyperP_best['SVC'] = svc_search.best_params_
        self.score['SVC'] = svc_search.best_score_

        rf = RandomForestClassifier()
        rf_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=self.hyperP_search['RF'],
            scoring=self.metric,
            n_iter=500,
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1)
        rf_search.fit(self.X, self.y)
        self.hyperP_best['RF'] = rf_search.best_params_
        self.score['RF'] = rf_search.best_score_

        xgb = XGBClassifier()
        xgb_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=self.hyperP_search['XGB'],
            scoring=self.metric,
            n_iter=500,
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        xgb_search.fit(self.X, self.y)
        self.hyperP_best['XGB'] = xgb_search.best_params_
        self.score['XGB'] = xgb_search.best_score_

        logit = LogisticRegression()
        logit_search = RandomizedSearchCV(
            estimator=logit,
            param_distributions=self.hyperP_search['LG'],
            scoring=self.metric,
            n_iter=500,
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        logit_search.fit(self.X, self.y)
        self.hyperP_best['LG'] = logit_search.best_params_
        self.score['LG'] = logit_search.best_score_

    def fit_predict_best_model(self, X_test):
        '''
        Find the best model
        Fit the whole training date with the best model
        '''
        if max(self.score) == 'RF':
            rf = RandomForestClassifier(
                n_estimators=self.hyperP_best['RF']['n_estimators'],
                min_samples_split=self.hyperP_best['RF']['min_samples_split'],
                min_samples_leaf=self.hyperP_best['RF']['min_samples_leaf'],
                max_depth=self.hyperP_best['RF']['max_depth'],
                bootstrap=self.hyperP_best['RF']['bootstrap'],
                n_jobs=-1,
                random_state=42
            )
            self.best_model = rf.fit(self.X, self.y)
            self.predictions = rf.predict(X_test)

        elif max(self.score) == 'SVC':
            svc = SVC(
                C=self.hyperP_best['SVC']['C'],
                random_state=42
            )

            self.best_model = svc.fit(self.X_svc, self.y_svc)
            self.predictions = svc.predict(X_test)

        elif max(self.score) == 'XGB':
            xgb = XGBClassifier(
                n_estimators=self.hyperP_best['XGB']['n_estimators'],
                subsample=self.hyperP_best['XGB']['subsample'],
                min_child_weight=self.hyperP_best['XGB']['min_child_weight'],
                max_depth=self.hyperP_best['XGB']['max_depth'],
                gamma=self.hyperP_best['XGB']['gamma'],
                eta=self.hyperP_best['XGB']['eta'],
                colsample_bytree=self.hyperP_best['XGB']['colsample_bytree']
            )

            self.best_model = xgb.fit(self.X, self.y)
            self.predictions = xgb.predict(X_test)

        elif max(self.score) == 'KNN':
            knn = KNeighborsClassifier(
                n_neighbors=self.hyperP_best['KNN']['n_neighbors'],
                metric=self.hyperP_best['KNN']['metric']
            )

            self.best_model = knn.fit(self.X, self.y)
            self.predictions = knn.predict(X_test)

        else:
            logit = LogisticRegression(
                penalty=self.hyperP_best['LG']['penalty'])
            self.best_model = logit.fit(self.X, self.y)
            self.predictions = logit.predict(X_test)
