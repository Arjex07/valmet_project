# This app should be launched with `streamlit run valmet_app.py`
import streamlit as st
import os
import sys
from datetime import datetime

# Asegura que Python encuentre valmet_analisis.py
# Añadimos la ruta absoluta para mayor robustez, aunque 'sys.path.append('.')' ya ayuda
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.append(current_dir)
icon_path = os.path.join(current_dir, "icon.png")

try:
    from valmet_analisis import run_analysis
except ImportError:
    st.error("Error: No se pudo importar 'valmet_analisis.py'. Asegúrate de que el archivo exista en la misma carpeta.")
    st.stop() # Detiene la ejecución de la app si no se puede importar la función


# --- Configuración de la App Streamlit ---
st.set_page_config(
    page_title="VALMET-GUI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=icon_path
)

# Colores de la interfaz
BACKGROUND_COLOR = "#2F2740"
SIDEBAR_COLOR = "#140D40"
TEXT_COLOR = "#D9CDBF"
BUTTON_COLOR = "#594C3C"
ACCENT_COLOR = "#A6998F"
FILE_TEXT_COLOR = "#1A1A1A"

# Estilos personalizados para aplicar la paleta corporativa
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {BACKGROUND_COLOR};
            color: {TEXT_COLOR};
        }}
        div[data-testid="stSidebar"] {{
            background-color: {SIDEBAR_COLOR};
            color: {TEXT_COLOR};
        }}
        div[data-testid="stSidebar"] .stButton>button {{
            background-color: {BUTTON_COLOR};
            color: {TEXT_COLOR};
        }}
        div[data-testid="stSidebar"] .stFileUploader span {{
            color: {FILE_TEXT_COLOR};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {ACCENT_COLOR};
        }}
        /* Input and file uploader colors */
        div[data-testid="stTextInput"] input {
            background-color: {ACCENT_COLOR};
            color: {SIDEBAR_COLOR};
        }
        div[data-testid="stFileUploader"] section label {
            background-color: {ACCENT_COLOR};
            color: {SIDEBAR_COLOR};
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Muestra el icono en la propia interfaz
#
# Paletas de colores disponibles (las mismas que usas)
PALETTES = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "coolwarm", "Spectral", "jet", "cubehelix", "terrain"
]

# --- Título y Descripción ---
st.markdown(
    """
    <h1 style='text-align:center;'>📈 VALMET-GUI</h1>
    <h3 style='text-align:center;'>Validación Automática de Lecturas de METeorología</h3>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Sube tu archivo CSV de datos meteorológicos horarios para generar un conjunto completo de gráficos y un informe resumen."
)

# --- Controles de Entrada (SideBar) ---
with st.sidebar:
    st.image(icon_path, width=150)
    st.header("Configuración del Análisis")
    st.markdown(
        """
        **Instrucciones**
        1. Sube un archivo CSV con las columnas `date`, `ws`, `wd` y `temp_K`.
        2. Ingresa un título descriptivo para tu proyecto.
        3. Selecciona la paleta de colores que prefieras.
        4. Pulsa **Generar Análisis** para procesar los datos.
        """
    )
    file_input = st.file_uploader(
        "Arrastra y suelta el archivo aquí\nLímite 200MB por archivo • CSV",
        type=["csv"],
        help="1. Sube tu archivo CSV (requiere columnas 'date', 'ws', 'wd', 'temp_K' y opcionalmente 'mixing_height_m')"
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

    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] img {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        div[data-testid="stSidebar"] .stButton > button {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
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
            temp_data_dir = os.path.join(current_dir, "temp_data_upload")
            try:
                # --- Guarda el archivo subido temporalmente ---
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

                if zip_file_path:
                    st.session_state.zip_download_path = zip_file_path
                    st.session_state.summary_html_content = summary_html_content
                    st.session_state.analysis_run_completed = True
                    st.success("Análisis completado exitosamente! 🎉")
                else:
                    st.session_state.analysis_error = (
                        "La función run_analysis no generó un archivo ZIP de resultados. "
                        "Por favor, verifica que los datos de entrada sean válidos."
                    )

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
    st.markdown(
        f"<p style='color:{TEXT_COLOR};'>Presiona 'Generar Análisis' en la barra lateral para comenzar. ➡️</p>",
        unsafe_allow_html=True,
    )

with st.expander("Tipos de Gráficos y Reportes Generados"):
    st.markdown(
        """
        - **Series Temporales:** Velocidad del viento, dirección del viento, temperatura y altura de mezcla.
        - **Ciclos Diarios:** Velocidad del viento, temperatura y altura de mezcla, mostrando promedio y rangos P5-P95.
        - **Rosas de los Vientos:** Se generan múltiples rosas del viento para las diferentes direcciones y velocidades.
        - **Mapa de Calor (Dirección vs. Hora):** Frecuencia de la dirección del viento por hora del día.
        - **Mapas de Calor Anuales:** Promedios anuales por hora del día y día del año para velocidad del viento, dirección del viento y temperatura.
        - **Vectores Promedio del Viento:** Vectores de viento promedio por hora para cada mes.
        - **Tablas Resumen:** Estadísticas clave (media, mediana, min, max, P5, P95, desviación estándar, cantidad de datos válidos) por variable.
        """
    )
