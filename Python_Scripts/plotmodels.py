import pandas as pd
import matplotlib.pyplot as plt

# --- Cargar datos ---
ts = pd.read_csv("ts_clean.csv", index_col=0, parse_dates=True)
df_arima = pd.read_csv("arima_forecast_2.csv", index_col=0, parse_dates=True)
df_sarima = pd.read_csv("sarimax_forecast_2.csv", index_col=0, parse_dates=True)
df_prophet = pd.read_csv("prophet_forecast_2.csv", index_col=0, parse_dates=True)
df_LGB = pd.read_csv('ts_LGB.csv', delimiter = ';' , index_col=0, parse_dates=True)
df_RF = pd.read_csv('ts_RF.csv', delimiter = ';' , index_col=0, parse_dates=True)

df_RF = df_RF.groupby('Fecha de solicitud')['Prediccion_Numero_Ampliaciones_RF'].sum().reset_index()

#print(df_agrupado)
# --- Gráfica combinada ---
plt.figure(figsize=(12,6))
plt.plot(ts, label="Histórico", color="black")

# Predicciones
plt.plot(df_arima["pred"], label="ARIMA", linestyle="--", color="blue")
plt.plot(df_sarima["pred"], label="SARIMA", linestyle="--", color="green")
plt.plot(df_prophet["yhat"], label="Prophet", linestyle="--", color="red")
plt.plot(df_LGB["Prediccion_Numero_Ampliaciones"], label="LGB", linestyle="--", color="purple")
plt.plot(df_RF['Fecha de solicitud'], df_RF["Prediccion_Numero_Ampliaciones_RF"], label="Random Forest", linestyle="--", color="yellow")

# Intervalos de confianza (opcional)
plt.fill_between(df_arima.index, df_arima["lower"], df_arima["upper"], color="blue", alpha=0.1)
plt.fill_between(df_sarima.index, df_sarima["lower"], df_sarima["upper"], color="green", alpha=0.1)
plt.fill_between(df_prophet.index, df_prophet["yhat_lower"], df_prophet["yhat_upper"], color="red", alpha=0.1)


plt.title("Comparación de modelos de predicción - ARIMA vs SARIMA vs Prophet vs LGB vs Random Forest")
plt.xlabel("Fecha")
plt.ylabel("Número de ampliaciones")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
