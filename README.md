# Heart Disease Classifier (DevOps Project)

A machine-learning web app that predicts whether a patient is likely to have heart disease from 13 clinical
values. The model is trained automatically, tested, packaged in Docker and deployed to AWS by a CI/CD pipeline.

**Flow:** Git/GitHub -> GitHub Actions (train -> test -> build -> push) -> Docker Hub -> AWS EC2.
Jenkinsfile, Kubernetes and Ansible files are included as extras.

## Problem Statement
Hospitals and clinics need quick screening support, but ML models are often stuck on a data scientist's laptop.
This project shows how to ship a heart disease model as a reliable, automatically deployed service.

## Dataset and Model
* Data: UCI Cleveland Heart Disease dataset (303 patients, 13 features) in `data/heart.csv`.
* Model: StandardScaler + Logistic Regression (scikit-learn), 80/20 train/test split.
* Test accuracy about 85% (precision 0.85, recall 0.82) on 61 held-out patients.
* Note: in this CSV the `target` column is reversed (1 = healthy), so `ml/train.py` flips it.
* **Educational project only - not a medical device or medical advice.**

## Structure
```
app/            Flask app (app.py, templates/index.html)
ml/train.py     Trains model -> model/model.pkl + metrics.json
data/heart.csv  Dataset
tests/          pytest tests (API tests + accuracy >= 75% gate)
Dockerfile      Trains the model during build, then serves with gunicorn
.github/workflows/ci-cd.yml   Main CI/CD pipeline
Jenkinsfile     Alternative pipeline
k8s/            Kubernetes Deployment + Service
ansible/        Ansible playbook for EC2
docs/architecture.png
```

## Run locally
```bash
pip install -r requirements.txt
python ml/train.py           # creates model/model.pkl
python -m pytest tests -v
python app/app.py            # http://localhost:5000
```

## Run with Docker
```bash
docker build -t heart-classifier .
docker run -d -p 80:5000 --name heart-classifier heart-classifier
docker ps                    # <- screenshot for assignment
```

## Test the API
```bash
curl -X POST http://localhost/predict -H "Content-Type: application/json" \
 -d '{"age":62,"sex":1,"cp":0,"trestbps":150,"chol":280,"fbs":1,"restecg":0,"thalach":105,"exang":1,"oldpeak":3.0,"slope":1,"ca":3,"thal":3}'
```

## Deploy on AWS
1. EC2 -> Launch instance: Ubuntu 22.04, t2.micro. Security Group: port 22 (My IP) and 80 (anywhere). Save the .pem key.
2. SSH in: `sudo apt update && sudo apt install -y docker.io && sudo usermod -aG docker ubuntu` (then reconnect).
3. Create a Docker Hub account + access token.
4. GitHub repo -> Settings -> Secrets and variables -> Actions, add:
   `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `EC2_HOST`, `EC2_USER` (ubuntu), `EC2_SSH_KEY` (full .pem text).
5. Push to `main`. Open `http://<EC2_PUBLIC_IP>`.

## Optional: Kubernetes / Ansible
```bash
# set YOUR_DOCKERHUB_USERNAME in k8s/deployment.yml and ansible/deploy.yml first
kubectl apply -f k8s/
ansible-playbook -i ansible/inventory.ini ansible/deploy.yml
```
