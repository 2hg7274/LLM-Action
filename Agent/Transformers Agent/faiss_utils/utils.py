import os
import json
import torch
import faiss
import numpy as np
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from configs import EMBEDDING_MODEL_PATH

def load_documents(file_paths):
    """
    주어진 파일 경로에서 문서를 읽고, 문단별로 분할합니다.
    """
    texts = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                texts.extend(paragraphs)
        else:
            print(f"File {file_path} does not exist.")
    return texts


def create_dataset_from_files(file_paths):
    """
    파일 경로 리스트로부터 문서를 불러와 HuggingFace Dataset을 생성합니다.
    """
    texts = load_documents(file_paths)
    data = {"text": texts}
    return Dataset.from_dict(data)


def load_embedding_model_and_tokenizer(model_name: str = EMBEDDING_MODEL_PATH):
    """
    벡터 검색을 위한 임베딩 모델과 토크나이저를 로드합니다.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def compute_embeddings_for_dataset(dataset: Dataset, tokenizer, model) -> Dataset:
    """
    데이터셋의 각 텍스트 항목에 대해 임베딩을 계산하고, "embeddings" 컬럼으로 추가합니다.
    """
    def compute_embeddings(batch):
        encoded = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = model(**encoded)
        embeddings = model_output.last_hidden_state.mean(dim=1)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        batch["embeddings"] = embeddings.cpu().numpy().tolist()
        return batch
    return dataset.map(compute_embeddings, batched=True)


def build_faiss_index(dataset: Dataset, column: str = "embeddings") -> Dataset:
    """
    데이터셋의 지정된 컬럼에 대해 Faiss 인덱스를 구축합니다.
    """
    dataset.add_faiss_index(column=column)
    print(f"Faiss 인덱스가 {dataset.num_rows}개의 문단으로 구축되었습니다.")
    return dataset


def get_query_embedding(query, tokenizer, model) -> np.ndarray:
    """
    질의 텍스트에 대한 임베딩을 계산하여 반환합니다.
    """
    if isinstance(query, dict):
        query = json.dumps(query, sort_keys=True)
    encoded_query = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        query_output = model(**encoded_query)
    query_embedding = query_output.last_hidden_state.mean(dim=1)
    query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
    return query_embedding.cpu().numpy()


def search_similar_documents_faiss(query_text, dataset, tokenizer, model, k=2):
    """
    Faiss 라이브러리를 직접 사용하여 유사 문서를 검색하는 함수입니다.
    이미 저장된 "faiss_index.index" 파일이 있다면 이를 로드하고, 없으면 새로 생성 후 저장합니다.
    """
    # 질의 임베딩 계산
    query_embedding = get_query_embedding(query_text, tokenizer, model)  # shape: (1, d)
    
    # dataset에서 모든 임베딩을 numpy 배열로 추출 (shape: (N, d))
    embeddings = np.array(dataset["embeddings"])
    d = embeddings.shape[1]
    
    index_path = "./faiss_index.index"
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"Faiss 인덱스를 {index_path} 파일에서 로드했습니다.")
    else:
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        faiss.write_index(index, index_path)
        print(f"Faiss 인덱스를 새로 생성하여 {index_path} 파일에 저장했습니다.")
    
    distances, indices = index.search(query_embedding, k)
    similar_texts = [dataset["text"][i] for i in indices[0]]
    return " ".join(similar_texts)