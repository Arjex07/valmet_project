#!/bin/bash

# Ruta absoluta a tu proyecto. Asegúrate de que esta ruta sea correcta.
# Puedes obtenerla escribiendo 'pwd' en tu terminal cuando estés en la carpeta 'valmet_project'.
PROJECT_DIR="/Users/josechavezperez/Documents/valmet_project"

# Cambia al directorio del proyecto
cd "$PROJECT_DIR"

# Activa tu entorno virtual donde Streamlit está instalado.
# Usamos ./.venv_streamlit porque es el nombre de la carpeta de tu entorno virtual.
source "./.venv_streamlit/bin/activate"

# Ejecuta la aplicación Streamlit
streamlit run valmet_app.py