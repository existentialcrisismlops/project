# project
## Project Execution (GitHub Action)
See project-execution.yml for building and executing the model.
## DVC Pipeline (dvc-pipeline.yml)
Defines a pipeline to create and push a new dataset version.
## Preprocess Script (preprocess.py)
Contains the preprocess function for data processing.
## Tasks Tracking (tasks.txt)
Lists tasks accomplished by each member.
## Google Auto-Authentication
Create a project on Google Cloud Console.
Enable Google Drive API.
Create credentials and obtain service account JSON.
Share Google Drive folder with service account.
Set up GitHub Actions secrets.
## MLflow Remote Server Setup (DAGsHub)
Create DAGsHub repository.
Connect to GitHub.
Add MLflow tracking URI to GitHub Actions secrets.
## Docker Deployment (Dockerfile) and Flask App
docker run -p 8080:8080 existentialcrisismlops/e12:latest
## Monitoring of Concept Drift (monitor.py)
Monitors concept drift. See project-execution.yml for execution.