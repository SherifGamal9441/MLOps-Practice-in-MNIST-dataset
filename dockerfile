# Dockerfile
FROM python:3.10

WORKDIR /app

# 1. Copy only requirements first
COPY requirements.txt .

# 2. Install dependencies (this will be cached if requirements.txt doesn't change)
RUN pip install --no-cache-dir -r requirements.txt

# install dvc and setup Google Driver secret
RUN pip install dvc[gdrive]
COPY .dvc/ .dvc/
RUN dvc remote modify myremote gdrive_client_secret $GDRIVE_CLIENT_SECRET --local

# 3. Now copy the rest of your project
COPY app.py .
COPY mnist_model.h5 .
COPY requirements.txt .

RUN dvc pull

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
