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

## Pruebas

Para ejecutar las pruebas automatizadas utiliza:

```bash
pytest tests/test_run_analysis.py
```
