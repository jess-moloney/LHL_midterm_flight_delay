import pandas
import sklearn
import xgboost
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class test_models:

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
            'svm' : SVR(),
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
        poly = PolynomialFeatures(degree = self.degree)
        self.X_train_poly = poly.fit_transform(self.X_train)
        self.X_test_poly = poly.fit_transform(self.X_test)
        return LinearRegression()

    
    def create_models(self, all = False):
        if all or self.model_names is None:
            self.active_models = self.models.values()
        else:
            for name in self.model_names:
                self.active_models.append(self.models[name])

    def split_scale(self, data, target = None):
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
        lst_return = []
        for model in self.active_models:
            lst_return.append(self.eval_model(self.fit_predict(model)))
            pass
        return pandas.DataFrame(lst_return, columns=['MAE','RMSE','R2','ADJR2'], dtype = float)

    def fit_predict(self, model, poly = False):
        if poly:
            y_pred = model.fit(self.X_train_poly, self.y_train).predict(self.X_test_poly)
        else:
            y_pred = model.fit(self.X_train, self.y_train).predict(self.X_test)

        return y_pred


    def eval_model(self, y_pred):
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        r2 = r2_score(self.y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(self.X_test) - 1) / (len(self.X_test) - len(self.X_test[0,:] - 1))

        return (mae, rmse, r2, adj_r2)