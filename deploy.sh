#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Set variables
PROJECT_ID="your-project-id"
IMAGE_NAME="gcs-image-search"
REGION="us-central1"
SERVICE_NAME="gcs-image-search-service"

# Build the Docker image
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME .

# Push the image to Google Container Registry
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT_ID"

echo "Deployment completed successfully!"

