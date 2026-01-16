# Cloud Run deployment (M23)

This project deploys the FastAPI app using Cloud Run and Artifact Registry.

## One-time setup (local)

```bash
gcloud services enable artifactregistry.googleapis.com run.googleapis.com
gcloud artifacts repositories create cds-images \
  --repository-format=docker \
  --location=europe-west1 \
  --description="CDS predictor images"
gcloud auth configure-docker europe-west1-docker.pkg.dev
```

## GitHub Actions secret

Create a service account with these roles:
- Artifact Registry Writer
- Cloud Run Admin
- Service Account User

Store its JSON key in GitHub Actions as `GCP_SA_KEY`.

## Manual deploy (optional)

```bash
IMAGE=europe-west1-docker.pkg.dev/cds-predictor/cds-images/cds-api:latest
docker build -f dockerfiles/api.dockerfile -t "$IMAGE" .
docker push "$IMAGE"
gcloud run deploy cds-api \
  --image "$IMAGE" \
  --region europe-west1 \
  --platform managed \
  --allow-unauthenticated
```

## Verify

Open the Cloud Run URL and check `/health`.
