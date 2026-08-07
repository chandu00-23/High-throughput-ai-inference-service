import random
from locust import HttpUser, task, between

PROMPTS_POOL = [
    "Explain the concept of neural networks in simple terms.",
    "Can you explain neural networks in simple terms?",
    "What is machine learning?",
    "How does artificial intelligence work?",
    "Write a short python snippet to calculate fibonacci numbers.",
    "Can you provide a python function for fibonacci series?"
]

SYSTEM_PROMPTS = [
    "You are a helpful software engineer.",
    "You are a helpful software engineer."
]

class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def test_chat_completions(self):
        prompt = random.choice(PROMPTS_POOL)
        system_prompt = random.choice(SYSTEM_PROMPTS)

        payload = {
            "model": "llama-3-8b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "stream": False
        }

        self.client.post("/v1/chat/completions", json=payload, name="/v1/chat/completions")
