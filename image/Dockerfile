FROM python:3.13-slim

# Set working directory
WORKDIR /app/src

# Install gcc, g++, and make using apt (dnf doesn't exist in Debian-based images)
RUN apt-get update && apt-get install -y gcc g++ make

# Upgrade pip to latest version
RUN python3 -m pip install --upgrade pip

# Install key Python dependencies first
RUN pip install pysqlite3-binary numpy pandas spacy

# Copy requirements file and install remaining dependencies
COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt

# Set environment variable
ENV IS_USING_IMAGE_RUNTIME=True

# Copy everything exactly as is to /app
COPY . .
COPY .env /app/.env

# Expose port your app uses
EXPOSE 8000

# Default command to run your app
CMD ["python", "src/api_app_handler.py"]