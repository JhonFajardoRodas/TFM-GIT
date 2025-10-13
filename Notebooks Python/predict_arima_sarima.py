import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, itertools
warnings.filterwarnings("ignore")
import prophet


# Crear un DataFrame de ejemplo
path_file = 'C:/Users/UX530/Desktop/TFM-GIT/Notebooks Python/modeloST_CTO_sin_outlier.csv'
df = pd.read_csv(path_file, sep=';', dayfirst=True)
print(df.head())

date_col = [c for c in df.columns if 'fecha' in c.lower() or 'solicitud' in c.lower()][0]
target_col = [c for c in df.columns if 'ampli' in c.lower() or 'numero' in c.lower() or 'número' in c.lower()][0]

df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

df = df.sort_values(date_col).reset_index(drop=True)
ts = df.set_index(date_col)[target_col].sort_index()
print("Rango:", ts.index.min(), ts.index.max(), "len:", len(ts))

# --- 2. Limpieza mínima ---
if ts.isna().sum() > 0 and ts.isna().sum() <= 0.1*len(ts):
    ts = ts.interpolate(method='time')
else:
    ts = ts.dropna()


# Forzar freq semanal si es apropiado (ajusta si tus datos no son semanales)
freq = pd.infer_freq(ts.index)
if freq is None:
    # detectar delta mayoritario
    deltas = ts.index.to_series().diff().dropna().value_counts()
    common = str(deltas.index[0])
    if '7 days' in common:
        freq = 'W'
    elif '30 days' in common or '31 days' in common:
        freq = 'M'
    else:
        freq = 'W'
    ts = ts.asfreq(freq).interpolate(method='time')

    # --- 3. Split train/test ---
split_date = pd.Timestamp("2024-06-30")
if ts.index.max() > split_date:
    train = ts[:split_date]
    test = ts[split_date + pd.Timedelta(days=1):]
else:
    split_idx = int(0.8*len(ts))
    train = ts.iloc[:split_idx]
    test = ts.iloc[split_idx:]

    # Guardar versiones limpias
ts.to_csv("ts_clean.csv")
train.to_csv("ts_train.csv")
test.to_csv("ts_test.csv")

# --- 4. ARIMA grid search sencillo y ajuste ---
from statsmodels.tsa.arima.model import ARIMA

p = range(0,4); d = range(0,3); q = range(0,4)
pdq = list(itertools.product(p,d,q))
best_aic = np.inf; best_order=None; best_res=None
for order in pdq:
    try:
        model = ARIMA(train, order=order, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(method_kwargs={"warn_convergence": False})
        if np.isfinite(res.aic) and res.aic < best_aic:
            best_aic = res.aic; best_order = order; best_res = res
            print("Mejor ARIMA ahora:", best_order, best_aic)
    except Exception as e:
        continue

print("Mejor ARIMA final:", best_order, best_aic)

if best_res is not None:
    arima_res = best_res
    print(arima_res.summary())
    horizon_weeks = 52
    fc = arima_res.get_forecast(steps=horizon_weeks)
    df_arima = pd.DataFrame({
        "fecha": pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), periods=horizon_weeks, freq=pd.infer_freq(ts.index)),
        "pred": fc.predicted_mean.values, 
        "lower": fc.conf_int().iloc[:,0].values, 
        "upper": fc.conf_int().iloc[:,1].values
    }).set_index("fecha")
    df_arima.to_csv("arima_forecast_2.csv")
    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(ts, label='Histórico')
    plt.plot(df_arima['pred'], linestyle='--', label='ARIMA pred')
    plt.fill_between(df_arima.index, df_arima['lower'], df_arima['upper'], alpha=0.2)
    plt.legend(); plt.title("ARIMA Forecast")
    plt.show()


# --- 5. SARIMAX (búsqueda rápida, estacionalidad anual para datos semanales) ---
from statsmodels.tsa.statespace.sarimax import SARIMAX
freq = pd.infer_freq(ts.index)
seasonal_period = 52 if freq and freq.upper().startswith('W') else 12

ps = range(0,3); ds = range(0,2); qs = range(0,3)
Ps = range(0,2); Ds = range(0,2); Qs = range(0,2)
best_aic = np.inf; best_sar=None; best_order=None; best_seasonal=None
for order in itertools.product(ps, ds, qs):
    for seasonal in itertools.product(Ps, Ds, Qs):
        try:
            model = SARIMAX(train, order=order, seasonal_order=(seasonal[0], seasonal[1], seasonal[2], seasonal_period),
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_aic = res.aic; best_sar = res; best_order=order; best_seasonal=seasonal
                print("Mejor SARIMAX:", best_order, best_seasonal, best_aic)
        except Exception:
            continue

print("Mejor SARIMAX final:", best_order, best_seasonal, best_aic)

if best_sar is not None:
    fc = best_sar.get_forecast(steps=52)
    pm = fc.predicted_mean; ci = fc.conf_int()
    df_sar = pd.DataFrame({
        "fecha": pd.date_range(start=train.index[-1] + pd.Timedelta(days=1), periods=52, freq=pd.infer_freq(ts.index)),
        "pred": pm.values, "lower": ci.iloc[:,0].values, "upper": ci.iloc[:,1].values
    }).set_index("fecha")
    df_sar.to_csv("sarimax_forecast_2.csv")

