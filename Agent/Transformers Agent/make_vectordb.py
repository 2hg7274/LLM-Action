# Step 1: Install the required libraries:
# pip install datasets transformers faiss-cpu

import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from configs import EMBEDDING_MODEL_PATH

# ---------------------------
# Document File Functions
# ---------------------------

def write_documents():
    """
    Write two documents as separate txt files.
    """
    error_doc = """# Error Alert Resolution Document

This document outlines a general approach for addressing system errors triggered by excessive memory usage. It provides guidance on the cause of such errors and the recommended remedial measures to help restore normal operation.

## Error Cause

The error is triggered when the system’s memory usage exceeds a critical threshold (typically above 95%). When memory utilization reaches this level, the system is at risk of performance degradation or failure, prompting an automated error alert.

## Recommended Remedial Actions

1. **Identify High Memory Consumers:**  
   Monitor system processes to identify those consuming significant memory resources. Ensure that the process generating the alert is not mistakenly targeted.

2. **Terminate the Secondary Python Process:**  
   Instead of stopping the currently running Python program that issued the alert, locate and terminate the Python process that is using the next highest amount of memory. This action helps reduce overall memory consumption without affecting the main monitoring or management process.

3. **Follow-Up Verification:**  
   After terminating the identified process, verify that memory usage returns to acceptable levels. Continue to monitor system performance and investigate underlying causes to prevent recurrence.

## Conclusion

By promptly addressing high memory usage and carefully managing process termination, this approach helps maintain system stability and prevents further errors. Adhering to these guidelines ensures that the resolution process is both effective and minimally disruptive.
"""
    wifi_doc = """# Network Stability Alert Resolution Document

This document provides guidelines for addressing network stability issues. Network instability can be caused by hardware issues, interference, or misconfigurations.

## Action Required

Please check the WiFi status. Ensure that the WiFi signal strength is adequate and that no interference is affecting connectivity. If the issue persists, consult your network administrator.

## Conclusion

Regular monitoring of network performance is essential to maintain stable connectivity and productivity. Following these steps can help mitigate network issues effectively.
"""
    with open("./documents/error_resolution.txt", "w", encoding="utf-8") as f:
        f.write(error_doc)
    with open("./documents/wifi_resolution.txt", "w", encoding="utf-8") as f:
        f.write(wifi_doc)
    print("Documents written to disk.")

def load_documents(file_paths):
    """
    Read documents from the given file paths and split them into paragraphs.
    """
    texts = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Split content into paragraphs (using double newline as separator)
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                texts.extend(paragraphs)
        else:
            print(f"File {file_path} does not exist.")
    return texts

def create_dataset_from_files(file_paths):
    """
    Create a HuggingFace Dataset from a list of file paths by loading the documents.
    """
    texts = load_documents(file_paths)
    data = {"text": texts}
    return Dataset.from_dict(data)

# ---------------------------
# Alarm Query Functions
# ---------------------------

def combine_alarm_predictions(alarm_list):
    """
    Combine multiple alarm prediction dictionaries into a single descriptive text.
    """
    combined_texts = []
    for alarm in alarm_list:
        text = (f"Timestamp: {alarm['timestamp']}, CPU usage: {alarm['cpu_usage_percent']}%, "
                f"Memory usage: {alarm['memory_usage_percent']}%, Disk usage: {alarm['disk_usage_percent']}%, "
                f"Confidence: {alarm['confidence']}, Error occurred: {alarm['error_occur']}.")
        combined_texts.append(text)
    return " ".join(combined_texts)

# ---------------------------
# Model and Embedding Functions
# ---------------------------

def load_model_and_tokenizer(model_name: str):
    """
    Load the tokenizer and model from the HuggingFace Hub.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def compute_embeddings_for_dataset(dataset: Dataset, tokenizer, model) -> Dataset:
    """
    Compute embeddings for each text entry in the dataset using mean pooling,
    and add them as a new column.
    """
    def compute_embeddings(batch):
        encoded = tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = model(**encoded)
        # Mean pooling on token embeddings
        embeddings = model_output.last_hidden_state.mean(dim=1)
        # Normalize embeddings for similarity search
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        batch["embeddings"] = embeddings.cpu().numpy().tolist()
        return batch

    return dataset.map(compute_embeddings, batched=True)

def build_faiss_index(dataset: Dataset, column: str = "embeddings") -> Dataset:
    """
    Build a Faiss index on the specified column of the dataset.
    """
    dataset.add_faiss_index(column=column)
    print(f"Faiss index built with {dataset.num_rows} paragraphs.")
    return dataset

def save_vector_db(dataset: Dataset, column: str = "embeddings", index_path: str = "./faiss_index.index"):
    """
    Save the Faiss index to a local file.
    """
    dataset.save_faiss_index(column, index_path)
    print(f"Faiss index saved locally at: {index_path}")

def get_query_embedding(query, tokenizer, model) -> np.ndarray:
    """
    Compute and return the embedding for a query.
    The query can be either a string or a dictionary.
    If it is a dictionary, it is converted to a JSON string.
    """
    if isinstance(query, dict):
        query = json.dumps(query, sort_keys=True)
    encoded_query = tokenizer(query, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        query_output = model(**encoded_query)
    query_embedding = query_output.last_hidden_state.mean(dim=1)
    query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
    return query_embedding.cpu().numpy()

def query_vector_database(dataset: Dataset, query_embedding: np.ndarray, k: int = 2):
    """
    Retrieve the top k most similar paragraphs from the vector database.
    """
    scores, retrieved_examples = dataset.get_nearest_examples("embeddings", query_embedding[0], k=k)
    return scores, retrieved_examples

# ---------------------------
# Main Execution
# ---------------------------

def main():
    # Write out the documents as txt files.
    write_documents()

    # Define file paths for the documents
    file_paths = ["./documents/error_resolution.txt", "./documents/wifi_resolution.txt"]

    # Create dataset from the text files
    dataset = create_dataset_from_files(file_paths)
    print(f"Dataset created with {dataset.num_rows} paragraphs.")

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(EMBEDDING_MODEL_PATH)

    # Compute embeddings for the dataset
    dataset = compute_embeddings_for_dataset(dataset, tokenizer, model)

    # Build the Faiss index
    dataset = build_faiss_index(dataset)

    # Save the vector DB (Faiss index) locally
    save_vector_db(dataset, column="embeddings", index_path="./faiss_index.index")

    # Define alarm predictions as a list of dictionaries
    alarm_prediction = [
        {
            'timestamp': '2025-03-31T10:00:00',
            'cpu_usage_percent': 68.4,
            'memory_usage_percent': 82.6,
            'disk_usage_percent': 70.1,
            'confidence': 0.6,
            'error_occur': False
        },
        {
            'timestamp': '2025-03-31T10:05:00',
            'cpu_usage_percent': 75.9,
            'memory_usage_percent': 97.3,
            'disk_usage_percent': 85.7,
            'confidence': 0.95,
            'error_occur': True
        }
    ]

    # Combine all alarm predictions into one text query
    combined_query = combine_alarm_predictions(alarm_prediction)
    print("Combined Alarm Query Text:\n", combined_query)

    # Compute the query embedding using the combined alarm query text
    query_embedding = get_query_embedding(combined_query, tokenizer, model)

    # Retrieve the top 2 most similar paragraphs from the vector database
    scores, retrieved_examples = query_vector_database(dataset, query_embedding, k=2)

    print("\nQuery Results:")
    # retrieved_examples might be a dict with a "text" key
    if isinstance(retrieved_examples, dict):
        for text in retrieved_examples.get("text", []):
            print("\nParagraph:", text)
    else:
        for example in retrieved_examples:
            print("\nParagraph:", example["text"])

if __name__ == "__main__":
    main()
