import pandas as pd

import matplotlib.pyplot as plt

# Datos de ejemplo para el diagrama de Gantt
tasks = ['Investigación de la literatura del proyecto', 
        'Recopilación y preparación de los datos', 
        'Planteamiento del problema', 
        'Análisis de los datos y elaboración del Dashboard',
        'Exposición y evaluación de los resultados',
        'Conclusiones de los resultados'
        ]
start_dates = ['2025-07-01', '2025-08-01', '2025-08-15', '2025-10-01', '2025-10-15', '2025-10-15']
end_dates = ['2025-08-01', '2025-08-15', '2025-10-01', '2025-10-15', '2025-10-20', '2025-10-20']

# Convertir las fechas a formato datetime
start_dates = pd.to_datetime(start_dates)
end_dates = pd.to_datetime(end_dates)

# Crear un DataFrame
df = pd.DataFrame({
    'Tarea': tasks,
    'Inicio': start_dates,
    'Fin': end_dates
})

# Crear el diagrama de Gantt
fig, ax = plt.subplots(figsize=(18, 6))

for i, task in enumerate(df['Tarea']):
    ax.barh(task, (df['Fin'][i] - df['Inicio'][i]).days, left=df['Inicio'][i], color='skyblue')

# Configurar el formato del eje x
ax.set_xlabel('Fecha')
ax.set_ylabel('Tareas')
ax.set_title('Diagrama de Gantt')
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar el gráfico
plt.show()