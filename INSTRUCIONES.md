# Instrucciones para Ejecutar la Aplicación Streamlit VALMET

Este archivo contiene los pasos para iniciar y ejecutar la aplicación Streamlit de análisis VALMET.

---

## **Prerrequisitos**

Asegúrate de tener instalado Python y los paquetes necesarios (incluyendo Streamlit) en tu entorno virtual. Si no has configurado el entorno virtual, puedes hacerlo así (asegúrate de estar en la carpeta `valmet_project`):

```bash
python3 -m venv .venv_streamlit
source .venv_streamlit/bin/activate
pip install streamlit pandas matplotlib seaborn windrose
# Instala cualquier otra dependencia que necesites, por ejemplo:
# pip install openpyxl # Si trabajas con archivos .xlsx
---

## **Método de Ejecución Recomendado (Script)**

La forma más sencilla de iniciar la aplicación es usando el script `iniciar_app.sh` que ya hemos configurado.

1.  **Abre una Terminal** (en VS Code: `Terminal` > `New Terminal`).
2.  **Navega hasta la carpeta de tu proyecto `valmet_project`** si aún no estás allí. (Tu ruta es `/Users/josechavezperez/Documents/valmet_project`).

    ```bash
    cd /Users/josechavezperez/Documents/valmet_project
    ```

3.  **Asegúrate de que el script `iniciar_app.sh` sea ejecutable.** (Solo necesitas hacer esto una vez).

    ```bash
    chmod +x iniciar_app.sh
    ```

4.  **Ejecuta el script:**

    ```bash
    ./iniciar_app.sh
    ```

---