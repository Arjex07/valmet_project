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
pytest tests/test_run_analysis.py
```

## Empaquetado

Para distribuir la aplicación como un ejecutable independiente puedes usar
PyInstaller.

1. Instala PyInstaller:

```bash
pip install pyinstaller
```

2. Genera el ejecutable con:

```bash
pyinstaller --onefile --add-data "icon.png;." main.py
```

3. El archivo resultante se encuentra en `dist/main.exe`.

PyInstaller empaqueta todas las dependencias, por lo que quien reciba
`main.exe` no necesitará tener Python instalado para ejecutarlo.
