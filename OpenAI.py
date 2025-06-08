import os
from openai import OpenAI

client = OpenAI(
    # setx OPENROUTER_API_KEY "your_openAI_api_key"
    api_key=os.getenv("OPENAI_API_KEY")
)


def chat_with_standard(prompt: str, model: str = "gpt-4.1-nano") -> str:
    """標準モデルにチャットリクエストを送り、応答文字列を返す"""
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    try:
        return response.output[0].content[0].text
    except Exception:
        return "回答取得に失敗しました"


def chat_with_o3(prompt: str) -> str:
    """o4-mini モデルにチャットリクエストを送り、応答文字列を返す"""
    response = client.responses.create(
        model="o4-mini",
        reasoning={"effort": "low"},
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=10000,
    )
    try:
        return response.output[1].content[0].text
    except Exception:
        return "回答取得に失敗しました"


if __name__ == "__main__":
    prompt = "東京タワーの高さを教えてください。"
    answer = chat_with_standard(prompt)
    print(answer)
