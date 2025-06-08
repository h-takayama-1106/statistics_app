from openai import OpenAI
import os

openai_root = "openai/"  # OpenAI root directory
model_name = "o4-mini"  # 使用するモデル名
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    # powershell: setx OPENROUTER_API_KEY "your_openrouter_api_key"
    api_key=os.getenv("OPENROUTER_API_KEY"),  # APIキー
)

completion = client.chat.completions.create(
    model=model_name, messages=[{"role": "user", "content": "こんにちは"}]
)
print(completion)
print(completion.choices[0].message.content)
