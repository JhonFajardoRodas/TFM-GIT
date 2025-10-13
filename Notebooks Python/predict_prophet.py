#forecast.to_csv("arima_forecast_2.csv")

# ==========================================================
# PREDICCIÓN DE VOLUMEN CON PROPHET
# ==========================================================
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# ==========================================================
# 1️⃣ Cargar el dataset
# ==========================================================
df = pd.read_csv('C:/Users/UX530/Desktop/TFM-GIT/Notebooks Python/modeloST_CTO_sin_outlier.csv', sep=';')

# Renombrar las columnas al formato que Prophet necesita
df = df.rename(columns={
    "Fecha de solicitud": "ds",
    "Número de ampliaciones": "y"
})

# Convertir la columna de fecha a datetime
df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

# Quitar filas sin fecha válida
df = df.dropna(subset=["ds"])

# Ordenar por fecha
df = df.sort_values("ds")

print(df.head())
print(df.tail())

# ==========================================================
# 2️⃣ Definir fecha de corte y dividir datos
# ==========================================================
fecha_corte = "2024-06-30"

df_train = df[df["ds"] <= fecha_corte].copy()
df_test  = df[df["ds"] > fecha_corte].copy()

print(f"\nEntrenamiento hasta: {df_train['ds'].max().date()}")
if not df_test.empty:
    print(f"Predicción desde:    {df_test['ds'].min().date()}")
else:
    print("⚠️ No hay datos posteriores al 30/06/2025, se generará horizonte futuro manualmente.")

# ==========================================================
# 3️⃣ Crear y entrenar el modelo Prophet
# ==========================================================
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='additive'
)

m.fit(df_train)

# ==========================================================
# 4️⃣ Crear horizonte de predicción (manual si no hay test)
# ==========================================================
if df_test.empty:
    # Creamos fechas futuras manualmente (desde 07/07/2025 hasta 31/12/2025)
    future = pd.date_range(start="2025-07-07", end="2025-12-31", freq="W-MON")
    future = pd.DataFrame({"ds": future})
else:
    # Usar las fechas del test
    future = df_test[["ds"]].copy()

# ==========================================================
# 5️⃣ Generar la predicción
# ==========================================================
forecast = m.predict(future)

# ==========================================================
# 6️⃣ Graficar resultados
# ==========================================================
plt.figure(figsize=(12,6))
plt.plot(df_train["ds"], df_train["y"], label="Entrenamiento", color="black")

if not df_test.empty:
    plt.plot(df_test["ds"], df_test["y"], label="Real (Test)", color="gray", linestyle="--")

plt.plot(forecast["ds"], forecast["yhat"], label="Predicción Prophet", color="red")
plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                color="red", alpha=0.2, label="Intervalo de confianza")

plt.title("Predicción del Volumen de Ampliaciones con Prophet (desde 07/07/2025)")
plt.xlabel("Fecha")
plt.ylabel("Número de ampliaciones")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================================
# 7️⃣ Guardar resultados
# ==========================================================
forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
forecast_out.to_csv("prophet_forecast_2.csv", index=False)

print("✅ Predicciones guardadas en 'forecast_prophet.csv'")
