from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

HEART_DISEASE_UCIREPO_ID = 45


def prepare_data(normalize: bool = True):
    heart_disease = fetch_ucirepo(id=HEART_DISEASE_UCIREPO_ID)
    X = heart_disease.data.features
    vars_df = heart_disease.variables

    features = vars_df[vars_df["role"] == "Feature"]
    numeric_features = features[features["type"] == "Integer"]["name"].tolist()
    categorical_features = features[features["type"] == "Categorical"]["name"].tolist()

    y_original = heart_disease.data.targets.iloc[:, 0]
    y_binary = (y_original > 0).astype(int)

    if normalize:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train.to_numpy(), y_test.to_numpy()
