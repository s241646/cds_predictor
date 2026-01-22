from fastapi.testclient import TestClient
from cds_repository.api import app


def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "CDS Predictor API is live"


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()


def test_metrics_endpoint():
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "request_count" in response.json()


def test_predict_invalid_fasta():
    """Test that sending garbage text returns a 400 error."""
    with TestClient(app) as client:
        invalid_content = b"12345_NOT_A_SEQUENCE_DATA"
        files = {"fasta": ("invalid.txt", invalid_content, "text/plain")}
        response = client.post("/predict", files=files)
        assert response.status_code == 400
        assert "detail" in response.json()


def test_predict_valid_fasta():
    """Test a successful prediction with a real FASTA formatted string."""
    with TestClient(app) as client:
        # Standard FASTA format
        valid_fasta = ">seq1\nATGCATGCATGCATGCATGCATGCATGCATGC\n>seq2\nGCATGCATGCATGCATGCATGCATGCATGCAT\n"

        files = {"fasta": ("valid.fasta", valid_fasta.encode("utf-8"), "application/octet-stream")}
        # Sending form data for the optional flags
        data = {"return_logits": "true", "return_probs": "true", "batch_size": "16"}

        response = client.post("/predict", files=files, data=data)

        # If the model loaded correctly, this should be 200
        assert response.status_code == 200

        json_response = response.json()
        assert "results" in json_response
        assert len(json_response["results"]) == 2

        # Check structure of the first result
        first_result = json_response["results"][0]
        assert "id" in first_result
        assert "pred" in first_result
        assert "prob" in first_result  # Since we requested return_probs
        assert "logit" in first_result  # Since we requested return_logits


def test_config_endpoint():
    with TestClient(app) as client:
        response = client.get("/config")
        assert response.status_code == 200
        assert "max_batch" in response.json()
