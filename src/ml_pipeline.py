import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# Caminhos de entrada e saída
DATA_PATH = "data/processed/features_state_month.parquet"
MODEL_DIR = "data/models/"
OUTPUT_PATH = "data/processed/predictions.csv"
os.makedirs(MODEL_DIR, exist_ok=True)

print("🚀 Iniciando treinamento do modelo de previsão de queimadas...")

# ===============================
# 1️⃣ Carregar os dados processados
# ===============================
df = pd.read_parquet(DATA_PATH)
print(f"✅ Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")

# ===============================
# 2️⃣ Pré-processamento
# ===============================
# Codificar o estado (variável categórica)
le = LabelEncoder()
df["estado_encoded"] = le.fit_transform(df["estado"])

# Extrair ano e mês como números
df["ano"] = df["ano_mes"].str[:4].astype(int)
df["mes"] = df["ano_mes"].str[-2:].astype(int)

# Adicionar encoding temporal
df["mes_sin"] = np.sin(2 * np.pi * df["mes"]/12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes"]/12)
df["tempo"] = (df["ano"] - df["ano"].min()) * 12 + df["mes"]

# Atualizar X
X = df[["estado_encoded", "ano", "mes", "mes_sin", "mes_cos", "tempo", "focos"]]

# Features e target
X = df[["estado_encoded", "ano", "mes", "focos"]]
y = df["focos_next"]

# ===============================
# 3️⃣ Separar treino e teste
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"📊 Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

# ===============================
# 4️⃣ Treinar o modelo Random Forest
# ===============================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("🌲 Modelo RandomForest treinado com sucesso!")

# ===============================
# 5️⃣ Avaliação
# ===============================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📈 Avaliação do modelo:")
print(f"MAE  (Erro Médio Absoluto): {mae:.2f}")
print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.2f}")
print(f"R²   (Coeficiente de Determinação): {r2:.3f}")


# ===============================
# 6️⃣ Salvar modelo e previsões
# ===============================
# Salvar modelo
model_path = os.path.join(MODEL_DIR, "rf_model.joblib")
joblib.dump(model, model_path)
print(f"💾 Modelo salvo em: {model_path}")

# Gerar previsões completas
df["predicted_focos_next"] = model.predict(X)
df["erro_absoluto"] = abs(df["predicted_focos_next"] - df["focos_next"])

# Salvar previsões
df.to_csv(OUTPUT_PATH, index=False)
print(f"📄 Previsões salvas em: {OUTPUT_PATH}")

# Exibir amostra
print("\n🔍 Amostra das previsões:")
print(df[["estado", "ano_mes", "focos", "focos_next", "predicted_focos_next", "erro_absoluto"]].head())

print("\n🏁 Treinamento concluído com sucesso!")

# ===============================
# 7️⃣ Gerar previsões para meses futuros (2024–2025) com sazonalidade + tendência anual
# ===============================
import math

future_years = [2024, 2025]
future_months = list(range(1, 13))
future_rows = []

for estado in df["estado"].unique():
    for ano in future_years:
        for mes in future_months:
            future_rows.append({
                "estado": estado,
                "ano": ano,
                "mes": mes,
                "ano_mes": f"{ano}-{mes:02d}"
            })

df_future = pd.DataFrame(future_rows)
df_future["estado_encoded"] = le.transform(df_future["estado"])

# usar média de focos + modulação sazonal (sinusoidal)
mean_focos = df.groupby("estado")["focos"].mean().to_dict()
df_future["focos"] = df_future.apply(
    lambda row: mean_focos[row["estado"]] * (1 + 0.3 * math.sin((row["mes"] / 12) * 2 * math.pi)),
    axis=1
)

# previsões base
df_future["predicted_focos_next"] = model.predict(df_future[["estado_encoded","ano","mes","focos"]])
df_future["focos_next"] = np.nan
df_future["erro_absoluto"] = np.nan

# aplicar crescimento artificial (tendência temporal)
base_year = 2023
growth_rate = 0.03  # 3% ao ano
df_future["predicted_focos_next"] = df_future.apply(
    lambda r: r["predicted_focos_next"] * ((1 + growth_rate) ** (r["ano"] - base_year)),
    axis=1
)

# juntar com histórico
df_all = pd.concat([df, df_future], ignore_index=True)
df_all.to_csv("data/processed/predictions.csv", index=False)
print("📈 Futuras previsões (2024–2025) com sazonalidade e tendência salvas em data/processed/predictions.csv")