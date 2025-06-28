# 🧠 MNIST Classification – MLOps Practice Project

This is a hands-on MLOps practice project where I build, train, version, and deploy a TensorFlow-based CNN model on the MNIST dataset.

✅ I successfully deployed the model on the cloud using [Railway](https://railway.app).

---

## 🚀 Project Goals

- Practice end-to-end MLOps principles using a simple image classification task  
- Modularize code and track experiments  
- Use tools like Git, DVC, MLflow, Docker, and GitHub Actions  
- Deploy a FastAPI-powered inference API to the cloud  

---

## 🗂️ Project Structure

```
mlops-mnist/
├── .dvc/                     # DVC metadata
├── .github/                 # GitHub Actions for CI/CD
├── configs/                 # Configuration files (YAML)
├── data/                    # MNIST dataset (via DVC)
├── src/                     # Source code
│   ├── data/                # Data loading
│   ├── models/              # Model logic
│   └── utils/               # Utility functions
├── scripts/                 # CLI scripts
├── Dockerfile               # Docker image definition
├── dvc.yaml                 # DVC pipeline stages
├── requirements.txt         # Python dependencies
├── setup.py                 # CLI entry points and package config
├── .gitignore
├── .dockerignore
└── README.md
```

---

## ⚙️ Tools & Technologies

- **TensorFlow** – Model building  
- **DVC** – Data and model versioning  
- **MLflow** – Experiment tracking  
- **FastAPI** – REST API for inference  
- **Docker** – Containerization  
- **GitHub Actions** – CI/CD  
- **Railway** – Cloud deployment  

---

## 📦 Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Pull the dataset using DVC:**
   ```bash
   dvc pull
   ```

3. **Train the model:**
   ```bash
   mnist-train --epochs 5 --batch_size 64 --learning_rate 0.001
   ```

4. **Evaluate the model:**
   ```bash
   mnist-eval
   ```

5. **Start the FastAPI server:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

---

## 🖼️ Inference Example

You can send an image to your deployed endpoint using:

```bash
curl -X POST http://<your-url>/predict   -F "file=@/path/to/image.png"
```

---

## 📝 Notes

- It's meant for **learning and practice**, not production.  

---

## 🙋‍♂️ About Me

I'm a Self-Taught AI Engineer and aspiring MLOps who built this project to learn the full MLOps pipeline — from training and versioning to deployment and monitoring.

---

## 🏁 Next Steps

- Add Prometheus/Grafana for model monitoring  
- Set up data drift detection with [Evidently AI](https://evidentlyai.com/)  
- Extend to larger models or datasets