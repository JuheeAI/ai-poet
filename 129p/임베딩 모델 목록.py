import os
from dotenv import load_dotenv
import openai

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key)
models = client.models.list()

embedding_models = [model.id for model in models if "embed" in model.id]

print("\n--- 사용 가능한 임베딩 모델 목록 ---")
for m_id in sorted(embedding_models):
    print(f"- {m_id}")
