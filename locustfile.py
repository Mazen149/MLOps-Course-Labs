from locust import HttpUser, between, task


class ChurnPredictionUser(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(4)
    def predict_churn(self):
        payload = {
            "credit_score": 600,
            "age": 40,
            "tenure": 3,
            "balance": 60000.0,
            "num_of_products": 2,
            "has_cr_card": 1,
            "is_active_member": 1,
            "estimated_salary": 50000.0,
            "geography": "Spain",
            "gender": "Male",
        }

        self.client.post("/predict", json=payload)
