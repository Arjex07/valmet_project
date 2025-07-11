import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Eliminamos plotly.express y plotly.graph_objects si vamos a Matplotlib puro
# import plotly.express as px
# import plotly.io as pio
from windrose import WindroseAxes
import os
import zipfile
import glob
import datetime
import matplotlib.colors
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import MaxNLocator
import calendar 
# from plotly.subplots import make_subplots # No necesario si no usamos Plotly
# import plotly.graph_objects as go # No necesario si no usamos Plotly

# pio.templates.default = "plotly_white" # No necesario si no usamos Plotly

print("valmet_analisis.py: Script cargado.")

def celsius_from_kelvin(temp_k):
    """Convierte temperatura de Kelvin a Celsius."""
    return temp_k - 273.15

def create_output_directory(base_dir="VALMET_OUTPUT"):
    """Crea un directorio de salida único con marca de tiempo."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"analisis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"valmet_analisis.py: Directorio de salida creado: {output_dir}")
    return output_dir

def generate_time_series_plots(df, output_dir, palette_name, project_title):
    """Genera gráficos de series temporales para ws, wd, temp_C, mixing_height_m usando Matplotlib."""
    print("valmet_analisis.py: Generando gráficos de series temporales (Matplotlib)...")
    
    generated_files = [] 

    # Colores de la paleta para Matplotlib
    cmap = plt.cm.get_cmap(palette_name)

    variables_info = {
        'ws': 'Velocidad del Viento [m/s]',
        'wd': 'Dirección del Viento [°]',
        'temp_c': 'Temperatura [°C]',
        'mixing_height_m': 'Altura de Mezcla [m]'
    }

    for i, (var, label) in enumerate(variables_info.items()):
        if var in df.columns and not df[var].dropna().empty:
            plt.figure(figsize=(12, 4))
            plt.plot(df.index, df[var], lw=0.5, color=cmap(0.2 + i*0.1)) # Usar la paleta de colores
            plt.title(f"{project_title} - Serie Temporal de {label}")
            plt.xlabel("Fecha")
            plt.ylabel(label)
            plt.grid(True)
            
            # Formateo del eje X para fechas, similar a lo que hacía Plotly
            # Optar por AutoFormatter si hay muchos datos, o FixedLocator si se desea un control estricto.
            # Para Google Colab, era plt.grid(True) y matplotlib manejaba el formato.
            # Aquí, lo dejaremos más genérico para que Matplotlib lo adapte.
            plt.tight_layout()
            
            fig_path = os.path.join(output_dir, f"{project_title.replace(' ', '_')}-serie_{var}.png")
            plt.savefig(fig_path, dpi=300) # Guardar con buena resolución
            plt.close() # Cerrar la figura para liberar memoria
            print(f"valmet_analisis.py: Guardado {fig_path}")
            generated_files.append(fig_path)
        else:
            print(f"valmet_analisis.py: ADVERTENCIA: No hay datos suficientes para la serie temporal de {label}.")
    print("valmet_analisis.py: Gráficos de series temporales generados.")
    return generated_files

def generate_daily_cycles_plots(df, output_dir, palette_name, project_title):
    """Genera gráficos de ciclos diarios para ws, temp_C, mixing_height_m con P5-P95 (Matplotlib)."""
    print("valmet_analisis.py: Generando gráficos de ciclos diarios (Matplotlib)...")
    generated_files = [] 

    df['hour'] = df.index.hour # Asegurarse de que 'hour' exista
    variables_info = {
        'ws': 'Velocidad del Viento [m/s]',
        'temp_c': 'Temperatura [°C]',
        'mixing_height_m': 'Altura de Mezcla [m]'
    }

    for var, label in variables_info.items():
        if var in df.columns and not df[var].dropna().empty:
            grupo = df.groupby('hour')[var]
            media = grupo.mean()
            p5 = grupo.quantile(0.05)
            p95 = grupo.quantile(0.95)

            fig, ax = plt.subplots(figsize=(10, 4))
            cmap = plt.cm.get_cmap(palette_name)
            ax.plot(media.index, media, label='Promedio', lw=2, color=cmap(0.5))
            ax.fill_between(media.index, p5, p95, alpha=0.3, label='P5–P95', color=cmap(0.2))
            
            ax.set_title(f"{project_title} - Ciclo diario – {label}")
            ax.set_xlabel("Hora del día")
            ax.set_ylabel(label)
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            
            fig_path = os.path.join(output_dir, f"{project_title.replace(' ', '_')}-ciclo_{var}.png")
            plt.savefig(fig_path, dpi=300)
            plt.close()
            print(f"valmet_analisis.py: Guardado {fig_path}")
            generated_files.append(fig_path) 
        else:
            print(f"valmet_analisis.py: ADVERTENCIA: No hay datos suficientes para el ciclo diario de {label}.")
    print("valmet_analisis.py: Gráficos de ciclos diarios generados.")
    return generated_files 

def generate_wind_roses(df, output_dir, project_title):
    """Genera múltiples rosas de los vientos para el DataFrame usando WindroseAxes (Matplotlib),
    replicando la lógica y cantidad del código original de Colab."""
    print("valmet_analisis.py: Generando rosas de los vientos (WindroseAxes/Matplotlib)...")
    
    generated_files = [] 

    # Asegúrate de que las columnas sean numéricas y existan
    df['wd'] = pd.to_numeric(df['wd'], errors='coerce')
    df['ws'] = pd.to_numeric(df['ws'], errors='coerce')

    # Asegurarse de que 'hour' y 'month' existan para los filtros
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
    if 'month' not in df.columns:
        df['month'] = df.index.month

    if 'wd' not in df.columns or 'ws' not in df.columns or df[['wd', 'ws']].dropna().empty:
        print("valmet_analisis.py: ADVERTENCIA: No hay datos suficientes para generar rosas de los vientos.")
        return generated_files

    # Definición de periodos igual que en tu código original de Colab
    periodos = {
        "Mañana (07:00–14:00)": df[df['hour'].between(7, 14)],
        "Tarde (15:00–00:00)": df[(df['hour'] >= 15) | (df['hour'] == 0)],
        "Noche (01:00–06:00)": df[df['hour'].between(1, 6)],
        "Hora 00:00–06:00": df[df['hour'].between(0, 6)],
        "Hora 06:00–12:00": df[df['hour'].between(6, 12)],
        "Hora 12_00–18_00": df[df['hour'].between(12, 18)],
        "Hora 18_00–23:00": df[df['hour'].between(18, 23)],
        "Anual": df,
        "Verano (21-dic a 20-mar)": df[((df['month'] == 12) & (df.index.day >= 21)) | (df['month'].isin([1, 2])) | ((df['month'] == 3) & (df.index.day <= 20))],
        "Invierno (21-jun a 22-sept)": df[((df['month'] == 6) & (df.index.day >= 21)) | (df['month'].isin([7, 8])) | ((df['month'] == 9) & (df.index.day <= 22))]
    }

    # Bins de velocidad del viento y etiquetas para la leyenda, como en tu código original
    bins = [0.5, 2.1, 3.6, 5.7, 8.8, 11.1, 100] # Añadimos un bin muy alto para asegurar que todo >= 11.1 caiga en el último bin
    labels = ["0.5–2.1", "2.1–3.6", "3.6–5.7", "5.7–8.8", "8.8–11.1", "≥11.1"]

    for nombre, subset in periodos.items():
        # Filtrar NaN en ws y wd para el subset actual
        subset_filtered = subset[['wd', 'ws']].dropna()
        if not subset_filtered.empty: 
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='windrose'))
            
            total = len(subset_filtered)
            calmas = len(subset_filtered[subset_filtered['ws'] < 0.5])
            pct_calma = f"Calma (<0.5 m/s): {calmas / total * 100:.1f}%" if total > 0 else "Calma (<0.5 m/s): 0.0%"
            
            # Usar 'bins' tal cual. WindroseAxes maneja internamente el último bin abierto.
            ax.bar(subset_filtered['wd'], subset_filtered['ws'], bins=bins, normed=True, calm_limit=0.5, opening=0.8, edgecolor='white')
            
            # Ajuste de la leyenda como en tu código original
            ax.set_legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=10, title="Velocidad [m/s]")
            
            # Acceder y modificar los textos de la leyenda para que coincidan con tus labels
            legend_texts = ax.get_legend().get_texts()
            for i, new_label in enumerate(labels):
                if i < len(legend_texts):
                    legend_texts[i].set_text(new_label)
            
            # Añadir el porcentaje de calma si es necesario, si WindroseAxes no lo hace por defecto
            # (WindroseAxes con calm_limit suele añadirlo automáticamente como la última entrada de la leyenda)
            if len(legend_texts) > len(labels): # Si hay un texto extra (el de calma)
                 legend_texts[len(labels)].set_text(pct_calma)
            else: # Si no lo añadió, lo podemos añadir como texto en el gráfico
                plt.text(0.5, -0.1, pct_calma, transform=ax.transAxes, ha='center', fontsize=9)


            fig.suptitle(f"{project_title} - Rosa del Viento ({nombre})", fontsize=13, y=1.08)
            # Asegurar nombres de archivo seguros
            safe_name = nombre.replace(' ', '_').replace(':', '_').replace('–', '-')
            nombre_img = os.path.join(output_dir, f"{project_title.replace(' ', '_')}-rosa_{safe_name.lower()}.png")
            fig.savefig(nombre_img, dpi=300, bbox_inches='tight')
            plt.close(fig)
            generated_files.append(nombre_img)
            print(f"valmet_analisis.py: Guardada rosa de los vientos: {nombre_img}")
        else:
            print(f"valmet_analisis.py: ADVERTENCIA: No hay datos válidos para rosa de viento: {nombre}")
    
    print("valmet_analisis.py: Rosas de los vientos generadas.")
    return generated_files

def generate_heatmap_direction_vs_hour(df, output_dir, project_title):
    """Genera un mapa de calor de la dirección del viento vs. la hora del día usando Seaborn."""
    print("valmet_analisis.py: Generando mapa de calor de dirección vs. hora (Seaborn)...")
    
    generated_files = [] 

    if 'wd' in df.columns and not df['wd'].dropna().empty:
        df_copy = df.copy() 
        df_copy['hour'] = df_copy.index.hour

        # Cuantificación de la dirección del viento en bins de 10 grados
        # Igual que en tu código original de Colab
        df_copy['wd_bin'] = (pd.to_numeric(df_copy['wd'], errors='coerce') // 10 * 10).astype('Int64')
        
        # Eliminar NaN antes de crosstab
        df_copy = df_copy.dropna(subset=['wd_bin', 'hour'])

        if not df_copy.empty:
            # Crea la tabla de contingencia y normaliza por columna para obtener porcentajes
            tabla1 = pd.crosstab(index=df_copy['wd_bin'], columns=df_copy['hour'], normalize='columns') * 100
            
            plt.figure(figsize=(12, 6))
            cmap = plt.cm.get_cmap('jet') # Usar 'jet' como en tu original
            levels = MaxNLocator(nbins=15).tick_values(0, tabla1.max().max())
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            
            sns.heatmap(tabla1, cmap=cmap, norm=norm, cbar_kws={'label': '%'})
            plt.title(f'{project_title} - Frecuencia Dirección vs Hora')
            plt.xlabel('Hora')
            plt.ylabel('Dirección [°]')
            plt.tight_layout()
            
            fig_path = os.path.join(output_dir, f"{project_title.replace(' ', '_')}-heatmap_dir_hora.png")
            plt.savefig(fig_path, dpi=300)
            plt.close()
            generated_files.append(fig_path)
            print(f"valmet_analisis.py: Guardado {fig_path}")
        else:
            print("valmet_analisis.py: ADVERTENCIA: No hay datos suficientes para heatmap Dirección vs Hora.")
    else:
        print("valmet_analisis.py: ADVERTENCIA: No hay columnas 'wd' válidas para heatmap Dirección vs Hora.")
    
    print("valmet_analisis.py: Mapa de calor de dirección vs. hora generado.")
    return generated_files

def generate_annual_heatmaps(df, output_dir, project_title):
    """Genera mapas de calor anuales para velocidad, temperatura y altura de mezcla usando Seaborn."""
    print("valmet_analisis.py: Generando mapas de calor anuales (Seaborn)...")
    
    generated_files = [] 

    # Asegurarse de que 'month' y 'hour' existan
    if 'month' not in df.columns:
        df['month'] = df.index.month
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
    
    variables_info = {
        'ws': {'label': 'Velocidad del Viento [m/s]', 'cmap': 'coolwarm'}, # Tu original usaba coolwarm
        'temp_c': {'label': 'Temperatura [°C]', 'cmap': 'coolwarm'}, # Tu original usaba coolwarm para temperatura
        'mixing_height_m': {'label': 'Altura de Mezcla [m]', 'cmap': 'coolwarm'} # Tu original usaba coolwarm
    }

    for var, info in variables_info.items():
        if var in df.columns and not df[var].dropna().empty:
            df_copy = df.copy() 
            df_copy = df_copy[df_copy[var].notnull()] # Filtrar NaNs
            
            if not df_copy.empty:
                # Pivote para obtener mes vs hora
                tabla = df_copy.groupby(['month', 'hour'])[var].mean().unstack()
                
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.heatmap(tabla, cmap=info['cmap'], cbar_kws={'label': info['label']}, ax=ax)
                
                ax.set_title(f"{project_title} - Ciclo Anual de {info['label']}")
                ax.set_xlabel("Hora del Día")
                ax.set_ylabel("Mes")
                ax.set_yticks(np.arange(0.5, 12.5, 1)) # Centrar las etiquetas
                ax.set_yticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
                plt.tight_layout()
                
                fig_path = os.path.join(output_dir, f"{project_title.replace(' ', '_')}-heatmap_anual_{var}.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                generated_files.append(fig_path)
                print(f"valmet_analisis.py: Guardado {fig_path}")
            else:
                print(f"valmet_analisis.py: ADVERTENCIA: No hay datos suficientes para heatmap anual de '{info['label']}'.")
        else:
            print(f"valmet_analisis.py: ADVERTENCIA: Columna '{var}' no válida o vacía para heatmap anual.")
    
    print("valmet_analisis.py: Mapas de calor anuales generados.")
    return generated_files

def generate_average_wind_vectors(df, output_dir, project_title):
    """Genera un gráfico de vectores de viento promedio por hora y mes usando Matplotlib."""
    print("valmet_analisis.py: Generando gráficos de vectores de viento promedio (Matplotlib)...")
    
    generated_files = [] 

    if 'wd' in df.columns and 'ws' in df.columns and not df[['wd', 'ws']].dropna().empty:
        df_copy = df.copy()
        df_copy['month'] = df_copy.index.month
        df_copy['hour'] = df_copy.index.hour

        # Calcular componentes u y v, como en tu código original
        df_copy['u'] = -pd.to_numeric(df_copy['ws'], errors='coerce') * np.sin(np.radians(pd.to_numeric(df_copy['wd'], errors='coerce')))
        df_copy['v'] = -pd.to_numeric(df_copy['ws'], errors='coerce') * np.cos(np.radians(pd.to_numeric(df_copy['wd'], errors='coerce')))
        
        # Eliminar NaN antes de agrupar
        df_copy = df_copy.dropna(subset=['u', 'v', 'ws', 'month', 'hour'])

        if not df_copy.empty:
            tabla_uv = df_copy.groupby(['month', 'hour']).agg({'u': 'mean', 'v': 'mean', 'ws': 'mean'}).reset_index()
            
            # Pivotear para obtener matrices de U, V y WS para todas las horas y meses
            all_months = range(1, 13)
            all_hours = range(0, 24)
            
            U = tabla_uv.pivot(index='month', columns='hour', values='u').reindex(index=all_months, columns=all_hours).fillna(0)
            V = tabla_uv.pivot(index='month', columns='hour', values='v').reindex(index=all_months, columns=all_hours).fillna(0)
            WS = tabla_uv.pivot(index='month', columns='hour', values='ws').reindex(index=all_months, columns=all_hours).fillna(0)
            
            X, Y = np.meshgrid(U.columns, U.index)
            
            plt.figure(figsize=(12, 6))
            cmap = plt.cm.get_cmap('jet') # Como en tu original
            norm = Normalize(vmin=0, vmax=WS.max().max())
            
            plt.contourf(X, Y, WS, cmap=cmap, norm=norm, levels=20) # Añadir levels para mejor gradiente
            plt.colorbar(label='Velocidad m/s')
            plt.quiver(X, Y, U, V, scale=50, color='k', width=0.0025) # Scale ajustado para mejor visualización
            
            plt.title(f'{project_title} - Vector Promedio del Viento por Hora y Mes')
            plt.xlabel('Hora del Día')
            plt.ylabel('Mes')
            plt.yticks(ticks=range(1,13), labels=['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'])
            plt.grid(True)
            plt.tight_layout()
            
            fig_path = os.path.join(output_dir, f"{project_title.replace(' ', '_')}-vector_viento.png")
            plt.savefig(fig_path, dpi=300)
            plt.close()
            generated_files.append(fig_path)
            print(f"valmet_analisis.py: Guardado {fig_path}")
        else:
            print("valmet_analisis.py: ADVERTENCIA: No hay datos suficientes para vector de viento promedio.")
    else:
        print("valmet_analisis.py: ADVERTENCIA: No hay columnas 'ws' o 'wd' válidas para vector de viento promedio.")
    
    print("valmet_analisis.py: Gráficos de vectores de viento promedio generados.")
    return generated_files 

def generate_summary_table(df, output_dir, project_title):
    """Genera una tabla resumen de estadísticas del DataFrame y la guarda como HTML."""
    print("valmet_analisis.py: Generando tabla resumen...")
    
    generated_files = [] 
    full_html_content = "" 

    if not df.empty:
        variables_estadisticas = {
            'ws': 'Velocidad del Viento [m/s]',
            'wd': 'Dirección del Viento [°]',
            'temp_k': 'Temperatura [K]',
            'mixing_height_m': 'Altura de Mezcla [m]'
        }

        estadisticas = []

        for var, nombre in variables_estadisticas.items():
            if var in df.columns and not df[var].dropna().empty:
                serie = pd.to_numeric(df[var], errors='coerce').dropna() 
                if var == 'temp_k':
                    serie = serie - 273.15 
                    nombre_display = 'Temperatura [°C]'
                else:
                    nombre_display = nombre

                if not serie.empty:
                    resumen = {
                        'Variable': nombre_display,
                        'Media': round(serie.mean(), 2),
                        'Mediana': round(serie.median(), 2),
                        'Mínimo': round(serie.min(), 2),
                        'Máximo': round(serie.max(), 2),
                        'Desviación estándar': round(serie.std(), 2),
                        'P5': round(serie.quantile(0.05), 2),
                        'P95': round(serie.quantile(0.95), 2),
                        'Cantidad de datos válidos': len(serie)
                    }
                    estadisticas.append(resumen)
                else:
                    print(f"valmet_analisis.py: ADVERTENCIA: No hay datos válidos para estadísticas de '{nombre_display}'.")

        if not estadisticas:
            print("valmet_analisis.py: No se pudieron generar estadísticas para ninguna variable.")
            full_html_content = "<h3>No se pudieron generar estadísticas de resumen.</h3>"
        else:
            df_estadisticas = pd.DataFrame(estadisticas)

            tabla_csv_path = os.path.join(output_dir, f"{project_title.replace(' ', '_')}_tabla_estadisticas.csv")
            df_estadisticas.to_csv(tabla_csv_path, index=False, sep=';', float_format='%.2f')
            print(f"valmet_analisis.py: Tabla de estadísticas generada y guardada en: {tabla_csv_path}")
            generated_files.append(tabla_csv_path) 

            summary_html = df_estadisticas.to_html(index=False, float_format='%.2f', classes='table table-striped')
            
            full_html_content = f"""
            <html>
            <head>
                <title>Resumen de Análisis - {project_title}</title>
                <style>
                    body {{ font-family: sans-serif; color: #333; }}
                    h2 {{ color: #2F2740; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9em; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .table-striped tbody tr:nth-of-type(odd) {{ background-color: rgba(0,0,0,.05); }}
                </style>
            </head>
            <body>
                <h2>Resumen Estadístico: {project_title}</h2>
                {summary_html}
                <p>Estadísticas calculadas sobre los datos válidos.</p>
            </body>
            </html>
            """
            html_file_path = os.path.join(output_dir, f"{project_title.replace(' ', '_')}-resumen_estadistico.html")
            with open(html_file_path, "w") as f:
                f.write(full_html_content)
            print(f"valmet_analisis.py: Guardado resumen estadístico en {html_file_path}")
            generated_files.append(html_file_path)

    else: 
        print("valmet_analisis.py: ADVERTENCIA: DataFrame vacío. No se pudo generar la tabla resumen.")
        full_html_content = "<h3>No hay datos disponibles para generar el resumen estadístico.</h3>"
    
    print("valmet_analisis.py: Tabla resumen generada.")
    return generated_files, full_html_content 

def create_zip_archive(output_dir, project_title, files_to_zip):
    """Comprime todos los gráficos y tablas en un archivo ZIP."""
    print("valmet_analisis.py: Creando archivo ZIP...")
    safe_project_title = project_title.replace(' ', '_').replace('.', '').replace('/', '_').replace('\\', '_')
    zip_filename = os.path.join(os.path.dirname(output_dir), f"{safe_project_title}_VALMET_Output.zip")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf: 
        for file_path in files_to_zip:
            if os.path.exists(file_path):
                arcname = os.path.join(os.path.basename(output_dir), os.path.basename(file_path))
                zipf.write(file_path, arcname)
            else:
                print(f"valmet_analisis.py: ADVERTENCIA: Archivo no encontrado para añadir al ZIP: {file_path}")
    
    print(f"valmet_analisis.py: Archivo ZIP creado: {zip_filename}")
    return zip_filename

def run_analysis(csv_file_path, output_dir, palette_name, project_title):
    print(f"valmet_analisis.py: Iniciando análisis para {csv_file_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    all_generated_files = [] 
    summary_html_content = "" 

    try:
        df = pd.read_csv(csv_file_path)
        df.columns = df.columns.str.strip().str.lower()
        print(f"valmet_analisis.py: Columnas después de limpieza: {df.columns.tolist()}")

        required_columns = ['date', 'ws', 'wd', 'temp_k'] 
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en el archivo CSV.")
        
        print("valmet_analisis.py: Procesando columna de fecha...")
        try:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
            df = df.set_index('date')
            df = df.sort_index()
        except Exception as e:
            raise ValueError(f"Error al procesar la columna 'date'. Asegúrate de que el formato sea DD-MM-AAAA HH:MM. Error: {e}")

        print("valmet_analisis.py: Convirtiendo columnas a numéricas...")
        if 'mixing_height_m' not in df.columns:
            df['mixing_height_m'] = np.nan
        numeric_cols = ['wd', 'ws', 'temp_k', 'mixing_height_m'] 
        
        for col in numeric_cols:
            if col in df.columns: 
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        initial_rows = len(df)
        df.dropna(subset=['ws', 'wd'], inplace=True)
        if len(df) < initial_rows:
            print(f"valmet_analisis.py: Se eliminaron {initial_rows - len(df)} filas con datos de viento faltantes.")

        if 'temp_k' in df.columns and not df['temp_k'].dropna().empty:
            df['temp_c'] = celsius_from_kelvin(df['temp_k'])
            print("valmet_analisis.py: Columna 'temp_c' calculada.")
        else:
            print("valmet_analisis.py: ADVERTENCIA: Columna 'temp_k' no encontrada o vacía, no se calculará 'temp_c'.")
            df['temp_c'] = np.nan 


        if not df.empty:
            # Tu código original de Colab puede haber filtrado por año o no.
            # Aquí, si hay múltiples años, filtramos por 2023 si existe, para replicar
            # lo que podría haber pasado si solo tenías datos de 2023 en Colab.
            # Si quieres analizar todos los años, elimina esta sección de filtrado.
            unique_years = df.index.year.unique()
            if len(unique_years) > 1 and 2023 in unique_years:
                df = df[df.index.year == 2023]
                print("valmet_analisis.py: Datos filtrados para el año 2023.")
            elif len(unique_years) > 1:
                print(f"valmet_analisis.py: Múltiples años ({unique_years.tolist()}) encontrados. Se analizarán todos los datos.")
            
            if df.empty:
                raise ValueError("No hay datos válidos después del preprocesamiento (ej. después de filtrar por año 2023 o eliminar NaNs).")
        else:
            raise ValueError("El DataFrame está vacío después de la lectura del CSV o el procesamiento inicial.")

        print("valmet_analisis.py: Comenzando generación de gráficos y resumen...")

        # --- Generación de Gráficos y Tablas ---
        all_generated_files.extend(generate_time_series_plots(df.copy(), output_dir, palette_name, project_title))
        all_generated_files.extend(generate_daily_cycles_plots(df.copy(), output_dir, palette_name, project_title))
        all_generated_files.extend(generate_wind_roses(df.copy(), output_dir, project_title)) # Genera múltiples rosas
        all_generated_files.extend(generate_heatmap_direction_vs_hour(df.copy(), output_dir, project_title))
        all_generated_files.extend(generate_annual_heatmaps(df.copy(), output_dir, project_title))
        all_generated_files.extend(generate_average_wind_vectors(df.copy(), output_dir, project_title))
        
        summary_files_list, summary_html_content = generate_summary_table(df.copy(), output_dir, project_title)
        all_generated_files.extend(summary_files_list)
        
        zip_file_path = create_zip_archive(output_dir, project_title, all_generated_files)

        print("valmet_analisis.py: Análisis completado.")
        return zip_file_path, summary_html_content

    except Exception as e:
        error_msg = f"Ocurrió un error inesperado durante el procesamiento o la generación: {e}. Por favor, revisa el formato de tus datos."
        print(error_msg)
        return None, None 

print("valmet_analisis.py: Script principal de análisis definido.")