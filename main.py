import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data():
    housing = fetch_openml(name="house_prices", as_frame=True, parser='auto')
    X = housing.data
    y = housing.target
    return X, y


def preprocess_data(X):
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns

    # Create transformers for categorical and numerical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Create a column transformer to apply transformations
    transformer = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ]
    )

    return transformer

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a preprocessing and modeling pipeline
    transformer = preprocess_data(X_train)
    model = Pipeline(steps=[('preprocessor', transformer),
                            ('regressor', LinearRegression())])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse

if __name__ == "__main__":
    X, y = load_data()

    with mlflow.start_run():
        model, mse = train_model(X, y)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")
