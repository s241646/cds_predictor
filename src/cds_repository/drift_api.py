# same file?

from fastapi import FastAPI, UploadFile
from google.cloud import storage
from data_drift import main
import os

app = FastAPI()

BUCKET = "cds-predictor"
INPUT_PREFIX = "uploaded_data"


@app.post("/drift/upload")
async def upload_and_run_drift(file: UploadFile):
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    safe_filename = os.path.basename(file.filename)
    blob_path = f"{INPUT_PREFIX}/{safe_filename}"

    blob = bucket.blob(blob_path)

    blob.upload_from_file(file.file)

    gcs_uri = f"gs://{BUCKET}/{blob_path}"

    # Run drift detection
    main(input_gcs_uri=gcs_uri, dataset_name="...")

    return {"status": "completed", "input": gcs_uri}


@app.post("/drift/run-latest")
def run_latest_drift():
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    blobs = list(bucket.list_blobs(prefix=INPUT_PREFIX))
    latest_blob = max(blobs, key=lambda b: b.updated)

    gcs_uri = f"gs://{BUCKET}/{latest_blob.name}"

    main(input_gcs_uri=gcs_uri, dataset_name="scheduled")

    return {"status": "scheduled_drift_completed", "input": gcs_uri}
