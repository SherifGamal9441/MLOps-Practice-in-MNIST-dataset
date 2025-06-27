# Dockerfile
FROM python:3.10

WORKDIR /app

# 1. Copy only requirements first
COPY requirements.txt .

# 2. Install dependencies (this will be cached if requirements.txt doesn't change)
RUN pip install --no-cache-dir -r requirements.txt

# 3. Now copy the rest of your project
COPY app.py .
COPY mnist_model.h5 .
COPY requirements.txt .


EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
