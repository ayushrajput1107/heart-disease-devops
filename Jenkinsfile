// Alternative CI/CD pipeline (Jenkins). Primary pipeline: .github/workflows/ci-cd.yml
pipeline {
    agent any
    environment { IMAGE = "YOUR_DOCKERHUB_USERNAME/heart-classifier" }
    stages {
        stage('Checkout') { steps { checkout scm } }
        stage('Train & Test') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'python ml/train.py'
                sh 'python -m pytest tests -v'
            }
        }
        stage('Build Docker Image') {
            steps { sh 'docker build -t $IMAGE:${BUILD_NUMBER} -t $IMAGE:latest .' }
        }
        stage('Push Image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub',
                                 usernameVariable: 'U', passwordVariable: 'P')]) {
                    sh 'echo $P | docker login -u $U --password-stdin'
                    sh 'docker push $IMAGE:latest'
                }
            }
        }
        stage('Deploy') {
            steps { sh 'ansible-playbook -i ansible/inventory.ini ansible/deploy.yml' }
        }
    }
}
