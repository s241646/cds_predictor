import random
import io
from locust import HttpUser, between, task


class CDSPredictorUser(HttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        """Executed when a simulated user starts."""
        self.fasta_content = self._generate_mock_fasta()

    def _generate_mock_fasta(self) -> bytes:
        """Creates a dummy FASTA file in memory."""
        seq = "".join(random.choice("ATGC") for _ in range(100))
        fasta_str = f">test_seq_{random.randint(1, 1000)}\n{seq}\n"
        return fasta_str.encode("utf-8")

    @task(1)
    def check_health(self):
        """Simulates checking if the API/Model is healthy."""
        self.client.get("/health")

    @task(1)
    def get_info(self):
        """Simulates checking model metadata."""
        self.client.get("/info")

    @task(5)
    def post_prediction(self):
        """
        The main load test: Uploading a FASTA file for prediction.
        """
        files = {"fasta": ("test.fasta", io.BytesIO(self.fasta_content), "application/octet-stream")}
        data = {"return_logits": "true", "return_probs": "true", "batch_size": "32"}

        with self.client.post("/predict", files=files, data=data, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                response.failure("Model not loaded (GCS issue?)")
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(1)
    def check_metrics(self):
        """Simulates a monitoring service hitting the metrics endpoint."""
        self.client.get("/metrics")
