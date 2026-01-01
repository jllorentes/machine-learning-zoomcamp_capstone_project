FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /workspace

# Copiar archivos de dependencias primero para aprovechar la cache
COPY pyproject.toml .
COPY uv.lock .

# Install uv
RUN pip install uv

# Install dependencies
# RUN uv pip install --system -r pyproject.toml
RUN uv pip install --system .

# Exponer puerto de Jupyter
EXPOSE 8888

# Comando por defecto: iniciar Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]