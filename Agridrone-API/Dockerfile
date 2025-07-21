FROM python:3.10

# RUN useradd -m -u 1000 user
# USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gdal-bin \
    python3-gdal \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python packages
COPY --chown=user ./requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ⬇️ Install the local modified Ultralytics library in editable mode
COPY --chown=user ./ultralytics ./ultralytics
RUN pip install -e ./ultralytics

# Copy all other files
COPY --chown=user . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
