# General Python Project Template

[English](#english) | [Español](#español)

---

<a name="english"></a>
## English

### About
This repository serves as a **robust and scalable template** for Python projects. It is designed to minimize setup time for data science and software development workflows by providing a pre-configured folder structure and modern dependency management tools.

**Repository Structure:**
* `app/`: Source code for the application (installable package).
* `data/`: Directory for datasets (raw and processed).
* `models/`: Storage for serialized models.
* `notebooks/`: Jupyter notebooks for experimentation and analysis.
* `docs/`: Project documentation.
* `logs/`: Application logs.

### Features
* **Modular Architecture:** The `app/` directory is configured as an editable package (`-e`), allowing you to import your own code easily into notebooks or scripts.
* **Modern Tooling:** Optimized for speed using `uv` for dependency resolution.
* **Data Science Ready:** Includes setup for Jupyter Kernels linked to the virtual environment.
* **Environment Management:** Clear instructions for `venv` creation.

### Usage
To start a new project using this structure:
1. Click the green **"Use this template"** button at the top right of this page.
2. Select **"Create a new repository"**.
3. Clone your new repo and follow the setup instructions below.

### Setup & Installation
Run the following commands to create your local environment and install dependencies:

```bash
python3.12 -m venv venv
source venv/bin/activate

# Install the app module in editable mode
pip install -e app/

# Install requirements using uv for speed
pip install uv
uv pip install -r requirements.txt

# Setup Jupyter Kernel
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```
### Dataset Source
Link: Insert Link Here

*Maintained by [MiquelGomezCorral](https://miquelgc.net)*

<a name="español"></a>
## Español

### Sobre el proyecto
Este repositorio sirve como una **plantilla robusta y escalable** para proyectos en Python. Está diseñado para minimizar el tiempo de configuración en flujos de trabajo de ciencia de datos y desarrollo de software, proporcionando una estructura de carpetas preconfigurada y herramientas modernas de gestión de dependencias.

**Estructura del Repositorio:**
* `app/`: Código fuente de la aplicación (paquete instalable).
* `data/`: Directorio para datasets (crudos y procesados).
* `models/`: Almacenamiento para modelos serializados.
* `notebooks/`: Jupyter notebooks para experimentación y análisis.
* `docs/`: Documentación del proyecto.
* `logs/`: Logs de la aplicación.

### Características
* **Arquitectura Modular:** El directorio `app/` está configurado como un paquete editable (`-e`), lo que permite importar tu propio código fácilmente en notebooks o scripts.
* **Herramientas Modernas:** Optimizado para velocidad usando `uv` para la resolución de dependencias.
* **Listo para Data Science:** Incluye configuración para Kernels de Jupyter vinculados al entorno virtual.
* **Gestión de Entorno:** Instrucciones claras para la creación de `venv`.

### Cómo usarlo
Para iniciar un nuevo proyecto usando esta estructura:
1. Haz clic en el botón verde **"Use this template"** (Usar esta plantilla) en la parte superior derecha de esta página.
2. Selecciona **"Create a new repository"** (Crear un nuevo repositorio).
3. Clona tu nuevo repo y sigue las instrucciones de configuración a continuación.

### Configuración e Instalación
Ejecuta los siguientes comandos para crear tu entorno local e instalar las dependencias:

```bash
python3.12 -m venv venv
source venv/bin/activate

# Instalar el módulo app en modo editable
pip install -e app/

# Instalar requisitos usando uv para mayor velocidad
pip install uv
uv pip install -r requirements.txt

# Configurar el Kernel de Jupyter
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```
### Fuente dataset
Link: Añade aquí el link a tu dataset


*Matenido por [MiquelGomezCorral](https://miquelgc.net)*
