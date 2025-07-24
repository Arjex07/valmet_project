# valmet_project
Aplicación Streamlit para Análisis de Datos Meteorológicos VALMET: Aplicación Streamlit para análisis y visualización de datos meteorológicos relevantes para la calidad del aire.

## Instalación

Para instalar las dependencias en un entorno virtual ejecuta:

```bash
python -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución de la aplicación

Lanza la interfaz de Streamlit con:

```bash
streamlit run valmet_app.py
```

La interfaz utiliza un tema oscuro personalizado. Si deseas cambiarlo puedes
editar los colores en `.streamlit/config.toml`.

## Pruebas

Para ejecutar las pruebas automatizadas utiliza:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

```bash
pytest tests/test_run_analysis.py
```

## Empaquetado

Para distribuir la aplicación como un ejecutable independiente puedes usar
PyInstaller. Ejecuta PyInstaller desde un entorno en el que tengas
`streamlit` instalado y añade la opción `--hidden-import=streamlit.web.cli`
para evitar el error `PackageNotFoundError`.

1. Instala PyInstaller:

```bash
pip install pyinstaller
```

2. Genera el ejecutable con:

El parámetro `--add-data` utiliza `;` como separador en Windows y `:` en
macOS/Linux.

```bash
pyinstaller --onefile --add-data "icon.png;." --hidden-import=streamlit.web.cli main.py
```

En macOS/Linux sustituye `;` por `:` en el argumento `--add-data`.

3. El archivo resultante se encuentra en `dist/main.exe` en Windows y en
`dist/main` en sistemas Unix.

PyInstaller empaqueta todas las dependencias, por lo que quien reciba el
ejecutable no necesitará tener Python instalado para ejecutarlo.
