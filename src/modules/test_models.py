import pandas
import sklearn
import xgboost
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              VotingRegressor)
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class test_models:
    """
    Test all sklearn + xgboost models against a dataset
    """
    def __init__ (self, data, target = None, model_names = None, max_depth = 2, degree = 2):
        self.data = data
        self.active_models = []
        self.max_depth = max_depth
        self.degree = degree
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_scale(data, target)
        self.model_names = model_names

        self.models = {
            'linear' : LinearRegression(),
            'ridge' : Ridge(),
            'lasso' : Lasso(),
            'polynomial' : self.polynomial_model(),
            'sgd' : SGDRegressor(),
            'svr' : SVR(),
            'd_tree' : DecisionTreeRegressor(),
            'r_forest' : RandomForestRegressor(max_depth = self.max_depth),
            'g_boost' : GradientBoostingRegressor(),
            'voting_r' : VotingRegressor(
                estimators=[
                    ('gb', GradientBoostingRegressor()),
                    ('rf', RandomForestRegressor()),
                    ('lr', LinearRegression())
                    ]
                ),
            'xgb' : xgboost.XGBRegressor(objective = 'reg:squarederror')
        }

    def polynomial_model(self):
        """
        Prepares data for a polynomial regression
        :returns:
        LinearRegression: Returns model for evaluation
        """
        poly = PolynomialFeatures(degree = self.degree)
        self.X_train_poly = poly.fit_transform(self.X_train)
        self.X_test_poly = poly.fit_transform(self.X_test)
        return LinearRegression()

    
    def create_models(self, all = False):
        """
        Creates list of active model for evaluation.
        The list comes from the model names given at initialization.
        :params:
        all (Bool): Adds all available models to the active list
        """
        if all or self.model_names is None:
            self.active_models = self.models.values()
        else:
            for name in self.model_names:
                self.active_models.append(self.models[name])

    def split_scale(self, data, target = None):
        """
        Splits and scales the data for the models
        :params:
        data : pd.DataFrame or np.Array to be used to split into the training and test sets.
        target : pd.DataFrame or np.Array used as target for model training

        :returns:
        X_train : training dataset
        X_test : testing dataset
        y_train : training target
        y_test : testing target
        """
        if target is None:
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data)
        else:
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target)

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def run_models(self):
        """
        Run all active models
        :returns:
        pd.DataFrame containing the Mean Absolute Error, Root Mean Squared Error, R^2 Score, and Adjusted R^2 Score
        for each active model
        """
        if self.model_names is None:
            index_label = list(self.models.keys())
        else:
            index_label = self.model_names
        lst_return = []
        for i, model in enumerate(self.active_models):
            if index_label[i] == 'polynomial':
                lst_return.append(self.eval_model(self.fit_predict(model, True)))
            else:
                lst_return.append(self.eval_model(self.fit_predict(model)))

        return pandas.DataFrame(lst_return, columns=['MAE','RMSE','R2','ADJR2'], index = index_label, dtype = float)

    def fit_predict(self, model, poly = False):
        """
        Fits each model on the training data then uses the testing set to predict the target
        :params:
        model : the active model to be fit and tested
        poly (Bool) : Normal dataset or dataset for polynomial regression
        :returns:
        y_pred : predicted target values
        """
        if poly:
            y_pred = model.fit(self.X_train_poly, self.y_train).predict(self.X_test_poly)
        else:
            y_pred = model.fit(self.X_train, self.y_train).predict(self.X_test)

        return y_pred


    def eval_model(self, y_pred):
        """
        Evaluate predicted target values against test target
        :params:
        y_pred : predicted target values
        :returns:
        Tuple of Mean Absolute Error, Root Mean Squared Error, R^2 Score, and Adjusted R^2 Score
        """
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        r2 = r2_score(self.y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(self.X_test) - 1) / (len(self.X_test) - len(self.X_test[0,:] - 1))

        return (mae, rmse, r2, adj_r2)
