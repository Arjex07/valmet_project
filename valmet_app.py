import streamlit as st
import os
import sys
from datetime import datetime
import shutil # Para manejar directorios de manera más robusta

# Asegura que Python encuentre valmet_analisis.py
# Añadimos la ruta absoluta para mayor robustez, aunque 'sys.path.append('.')' ya ayuda
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from valmet_analisis import run_analysis
except ImportError:
    st.error("Error: No se pudo importar 'valmet_analisis.py'. Asegúrate de que el archivo exista en la misma carpeta.")
    st.stop() # Detiene la ejecución de la app si no se puede importar la función


# --- Configuración de la App Streamlit ---
st.set_page_config(
    page_title="VALMET-GUI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paletas de colores disponibles (las mismas que usas)
PALETTES = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "coolwarm", "Spectral", "jet", "cubehelix", "terrain"
]

# --- Título y Descripción ---
st.markdown(
    """
    # 📈 VALMET-GUI
    ### *Validación Automática de Lecturas de METeorología*

    Sube tu archivo CSV de datos meteorológicos horarios para generar un completo conjunto de gráficos y un informe resumen.
    """
)

# --- Controles de Entrada (SideBar) ---
with st.sidebar:
    st.header("Configuración del Análisis")
    file_input = st.file_uploader(
        "1. Sube tu archivo CSV (requiere columnas 'date', 'ws', 'wd', 'temp_K' y opcionalmente 'mixing_height_m')",
        type=["csv"]
    )
    project_title_input = st.text_input(
        "2. Título para el Análisis",
        placeholder="Ej: Análisis Estación Maipo"
    )
    palette_dropdown = st.selectbox(
        "3. Elige una Paleta de Colores para los Gráficos",
        options=PALETTES,
        index=PALETTES.index("viridis") # Valor por defecto
    )

    run_button = st.button("🚀 Generar Análisis")

# --- Lógica del Botón y Salida ---
# Usamos un estado de sesión para mostrar los resultados solo después de que se generen
if 'analysis_run_completed' not in st.session_state:
    st.session_state.analysis_run_completed = False
if 'analysis_error' not in st.session_state:
    st.session_state.analysis_error = None
if 'zip_download_path' not in st.session_state:
    st.session_state.zip_download_path = None
if 'summary_html_content' not in st.session_state:
    st.session_state.summary_html_content = None


if run_button:
    st.session_state.analysis_run_completed = False # Reinicia el estado al presionar el botón
    st.session_state.analysis_error = None
    st.session_state.zip_download_path = None
    st.session_state.summary_html_content = None

    if file_input is not None and project_title_input:
        with st.spinner("Analizando datos y generando gráficos... Esto puede tardar unos minutos."):
            temp_csv_path = None
            output_directory = None
            try:
                # --- Guarda el archivo subido temporalmente ---
                temp_data_dir = os.path.join(current_dir, "temp_data_upload")
                os.makedirs(temp_data_dir, exist_ok=True)
                temp_csv_path = os.path.join(temp_data_dir, file_input.name)
                with open(temp_csv_path, "wb") as f:
                    f.write(file_input.getvalue())

                # --- Generar una carpeta de salida única para los resultados ---
                # Esta lógica ahora es similar a create_output_directory en valmet_analisis.py
                # Puedes considerar unificarla si quieres que valmet_analisis.py también maneje la creación de la base
                # Por ahora, la mantenemos aquí para que la app controle el directorio base.
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_results_base = os.path.join(current_dir, "VALMET_OUTPUT") # Usar el nombre base del módulo
                os.makedirs(output_results_base, exist_ok=True)
                output_directory = os.path.join(output_results_base, f"analisis_{timestamp}") # Consistente con el nombre de la función en valmet_analisis
                os.makedirs(output_directory, exist_ok=True) # Asegura que la carpeta exista

                # Llama a tu función de análisis desde valmet_analisis.py
                # La función run_analysis debe devolver la ruta del ZIP y el contenido HTML
                zip_file_path, summary_html_content = run_analysis(
                    temp_csv_path,
                    output_directory,
                    palette_dropdown,
                    project_title_input
                )

                st.session_state.zip_download_path = zip_file_path
                st.session_state.summary_html_content = summary_html_content
                st.session_state.analysis_run_completed = True
                st.success("Análisis completado exitosamente! 🎉")

            except Exception as e:
                st.session_state.analysis_error = f"Ocurrió un error durante el análisis: {e}"
                st.exception(e) # Muestra el traceback completo para depuración
            finally:
                # --- Limpiar archivos temporales ---
                if temp_csv_path and os.path.exists(temp_csv_path):
                    os.remove(temp_csv_path)
                # Opcional: limpiar la carpeta temp_data_upload si se quiere que siempre esté vacía
                if os.path.exists(temp_data_dir) and not os.listdir(temp_data_dir):
                    os.rmdir(temp_data_dir) 

    else:
        st.warning("Por favor, sube un archivo CSV y proporciona un título para el análisis. 🤔")


# --- Mostrar Resultados (fuera del bloque `if run_button` para persistir) ---
if st.session_state.analysis_run_completed:
    st.markdown("---")
    st.subheader("📊 Resultados del Análisis:")

    # Archivo ZIP descargable
    if st.session_state.zip_download_path and os.path.exists(st.session_state.zip_download_path):
        with open(st.session_state.zip_download_path, "rb") as f:
            st.download_button(
                label="⬇️ Descargar Todos los Gráficos y Reportes (ZIP)",
                data=f.read(),
                file_name=os.path.basename(st.session_state.zip_download_path),
                mime="application/zip"
            )
    else:
        st.error(f"No se encontró el archivo ZIP de resultados. Path: {st.session_state.zip_download_path} 😟")

    # Estadísticas resumen
    st.subheader("📝 Estadísticas Resumen:")
    if st.session_state.summary_html_content:
        st.components.v1.html(st.session_state.summary_html_content, height=400, scrolling=True)
    else:
        st.info("No hay resumen HTML disponible. ℹ️")

elif st.session_state.analysis_error:
    st.error(st.session_state.analysis_error)
else:
    # Mensaje inicial si no se ha ejecutado el análisis
    st.info("Presiona 'Generar Análisis' en la barra lateral para comenzar. ➡️")

st.markdown(
    """
    ---
    ### ✨ Tipos de Gráficos y Reportes Generados:
    - **Series Temporales:** Velocidad del viento, dirección del viento, temperatura y altura de mezcla.
    - **Ciclos Diarios:** Velocidad del viento, temperatura y altura de mezcla, mostrando promedio y rangos P5-P95.
    - **Rosa de los Vientos:** Una rosa de los vientos global para todas las direcciones y velocidades.
    - **Mapa de Calor (Dirección vs. Hora):** Frecuencia de la dirección del viento por hora del día.
    - **Mapas de Calor Anuales:** Promedios anuales por hora del día y día del año para velocidad del viento, dirección del viento y temperatura.
    - **Vectores Promedio del Viento:** Vectores de viento promedio por hora para cada mes.
    - **Tablas Resumen:** Estadísticas clave (media, mediana, min, max, P5, P95, desviación estándar, cantidad de datos válidos) por variable.
    """
)
