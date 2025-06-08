from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from OpenAI import chat_with_o3, chat_with_standard
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys

from statistics_functions import t_test, paired_t_test, simple_linear_regression, anova, kmeans_clustering  # type: ignore
from typing import List, Union, Dict


class Prompt(BaseModel):
    prompt: str
    model: str


class TTestRequest(BaseModel):
    group1: List[float]
    group2: List[float]
    equal_var: bool = False
    nan_policy: str = "omit"
    group1_name: str
    group2_name: str


app = FastAPI()

# CORS設定: フロントエンドからのリクエストを許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat_endpoint(request: Prompt):
    """プロンプトを受け取り、OpenAIへ送信し応答を返却"""
    # モデルによって呼び出す関数を切り替え
    if request.model == "o4-mini":
        response = chat_with_o3(request.prompt)
    else:
        response = chat_with_standard(request.prompt, request.model)
    return {"response": response}


@app.post("/paired_t_test")
async def paired_t_test_endpoint(request: TTestRequest):
    result = paired_t_test(
        request.group1,
        request.group2,
        nan_policy=request.nan_policy,
        group1_name=request.group1_name,
        group2_name=request.group2_name,
    )
    return result


@app.post("/t_test")
async def t_test_endpoint(request: TTestRequest):
    result = t_test(
        request.group1,
        request.group2,
        equal_var=request.equal_var,
        nan_policy=request.nan_policy,
        group1_name=request.group1_name,
        group2_name=request.group2_name,
    )
    return result


# --- Simple Linear Regression endpoint ------------------------------------
class SLRRequest(BaseModel):
    x: List[float]
    y: List[float]
    x_name: str
    y_name: str


@app.post("/simple_linear_regression")
async def simple_linear_regression_endpoint(request: SLRRequest):
    result = simple_linear_regression(
        request.x, request.y, x_name=request.x_name, y_name=request.y_name
    )
    return result


class ExplainRequest(BaseModel):
    method: str
    t_statistic: float
    p_value: float
    columns: List[str]


class SLRExplainRequest(BaseModel):
    slope: float
    intercept: float
    r_squared: float
    columns: List[str]


@app.post("/explain_slr")
async def explain_slr_endpoint(req: SLRExplainRequest):
    columns_str = ", ".join(req.columns)
    prompt = f"データの列: {columns_str}。単回帰分析の結果、傾き(slope)は{req.slope:.4f}、切片(intercept)は{req.intercept:.4f}、決定係数(R²)は{req.r_squared:.4f}です。この結果について専門用語を極力使わずに2行程度で端的に解説してください。"
    explanation = chat_with_standard(prompt, "gpt-4.1-nano")
    return {"explanation": explanation}


class AnovaRequest(BaseModel):
    groups: List[List[float]]
    nan_policy: str = "omit"


@app.post("/anova")
async def anova_endpoint(request: AnovaRequest):
    result = anova(*request.groups, nan_policy=request.nan_policy)
    return result


class KMeansRequest(BaseModel):
    data: List[List[float]]
    n_clusters: int = 3
    random_state: int = 0
    max_iter: int = 300
    group1_name: str
    group2_name: str


@app.post("/kmeans_clustering")
async def kmeans_clustering_endpoint(request: KMeansRequest):
    result = kmeans_clustering(
        request.data,
        n_clusters=request.n_clusters,
        random_state=request.random_state,
        max_iter=request.max_iter,
    )
    return result


@app.get("/")
async def root():
    """Serve the main application HTML."""
    return FileResponse("ホーム.html")


# Static file serving for files in this repository
app.mount("/static", StaticFiles(directory="."), name="static")


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """CSVファイルを受け取りサーバー側で解析し、列名とデータを返却"""
    content = await file.read()
    text = content.decode("utf-8")
    lines = text.strip().splitlines()
    if not lines:
        return {"message": "空のファイルです", "columns": [], "data": {}}
    header, *rows = lines
    cols = header.split(",")
    data: Dict[str, List[Union[float, str]]] = {col: [] for col in cols}
    for row in rows:
        if not row:
            continue
        values = row.split(",")
        for i, val in enumerate(values):
            try:
                num = float(val)
            except ValueError:
                num = val
            data[cols[i]].append(num)
    return {"message": f"read {len(rows)} rows", "columns": cols, "data": data}
