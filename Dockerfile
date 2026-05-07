FROM python:3.11-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY src/ ./src/

# Copy notebooks into subdirectory so ../src resolves to /app/src
COPY *.ipynb ./notebooks/

# Expose JupyterLab port
EXPOSE 8888

# Notebook working dir = /app/notebooks, so Path(cwd).parent = /app, and /app/src exists
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=${JUPYTER_TOKEN:-}", \
     "--ServerApp.root_dir=/app"]
