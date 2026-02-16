import mlflow
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from mlflow.models import infer_signature

test_df = pd.read_csv("sample_data/california_housing_test.csv")
train_df = pd.read_csv("sample_data/california_housing_train.csv")

y_train = train_df['median_house_value']
y_test = test_df['median_house_value']

x_train = train_df.drop(columns='median_house_value')
x_test = test_df.drop(columns='median_house_value')

# Configurar experimentos:

mlflow.set_experiment("Precos_Casas_California")
mlflow.sklearn.autolog() # loga parametros, metricas e modelos automaticamente

param_variations = [
    {"max_depth": 3, "min_samples_split": 5, "name": "Modelo_Conservador"},
    {"max_depth": 5, "min_samples_split": 10, "name": "Modelo_Equilibrado"},
    {"max_depth": 10, "min_samples_split": 2, "name": "Modelo_Complexo"}
]

for param_config in param_variations:

    run_name = param_config.pop('name')

    with mlflow.start_run(run_name=run_name) as run:

        print(f"Treinando: {run_name}...")

        # treino

        model = DecisionTreeRegressor(**param_config, random_state=42)
        model.fit(x_train, y_train)

        signature = infer_signature(x_train, model.predict(x_train))

        # registro de modelos

        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, run_name)

        