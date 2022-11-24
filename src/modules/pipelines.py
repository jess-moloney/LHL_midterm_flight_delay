class create_pipe:
    def __init__(self):
        self.data = data if droplist is None else data.drop(droplist, axis=1)
        self.categorical_features = data.select_dtypes(
            exclude="number"
        ).columns.tolist()
        self.quantitative_features = data.drop(
            self.categorical_features, axis=1
        ).columns.tolist()
        self.pipe

    def preprocessing(self):
        categorical_features = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                ("imputer", KNNImputer()),
            ]
        )

        quantitative_features = Pipeline(
            steps=[
                ("imputer", KNNImputer()),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("quantitative", quantitative_features, self.quantitative_features),
                ("categorical", categorical_features, self.categorical_features),
            ],
            sparse_threshold=0,
            remainder="drop",
        )

        return preprocessor

    def model(self):
        model = LinearRegression(random_state=constants.random_state)
        pipe = Pipeline(
            [
                ("preprocessor", self.build_preprocessor),
                ("dimensionality", FactorAnalysis(random_state=constants.random_state)),
                ("model", model),
            ]
        )
        self.pipe = pipe
        return self.pipe

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self.pipe

    def predict(self, X):
        return self.pipe.predict(X)

    def fit_predict(self, X_train, y_train, X_test):
        self.fit(X_train, y_train)
        return self.predict(X_test)

    def get_self(self):
        return self
