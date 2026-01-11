import logging
import json
import yaml
import tiktoken
from hashlib import md5

logger=logging.getLogger(__name__)
with open('config.yaml', 'r', encoding="utf-8") as file:
    config = yaml.safe_load(file)
DATASET_ROOT = config['dataset']['root'] # ="ckg_data/mix_chunk3"
DATASET = config['dataset']['dataset'] # ='mix'
MAX_TOKEN_SIZE = config['file_chunker']['max_token_size'] # =1024
OVERLAP_TOKEN_SIZE = config['file_chunker']['overlap_token_size'] # =128


MODEL = config['llm_provider']['model']
LLM_PROVIDER_API_KEY = config['llm_provider']['api_key']
LLM_PROVIDER_URL = config['llm_provider']['base_url']
LLM_PROVIDER_PORT = config['llm_provider0']['base_port']
EMBEDDING_MODEL = config['embedding_provider']['model']
EMBEDDING_URL = config['embedding_provider']['base_url']
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()

def chunk_documents(
    docs,
    model_name="cl100k_base",
    max_token_size=512,
    overlap_token_size=64,
):
    ENCODER = tiktoken.get_encoding(model_name)
    tokens_list = ENCODER.encode_batch(docs, num_threads=16)

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token_ids = []
        lengths = []

        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk = tokens[start : start + max_token_size]
            chunk_token_ids.append(chunk)
            lengths.append(len(chunk))

        # Decode all chunks
        chunk_texts = ENCODER.decode_batch(chunk_token_ids)

        for i, text in enumerate(chunk_texts):
            results.append({
                # "tokens": lengths[i],
                "hash_code": compute_mdhash_id(text),
                "text": text.strip().replace("\n", ""),
                # "chunk_order_index": i,
            })

    return results

def main(dataset_root: str, dataset: str, max_token_size: int=1024, overlap_token_size: int=128):
    ORIGINAL_TEXT_FILE=f"{dataset_root}/{dataset}.jsonl" # "datasets/mix/mix.jsonl"
    CHUNK_TEXT_FILE=f"{dataset_root}/{dataset}_chunk.json" # "datasets/mix/mix_chunk.json"

    data=[]
    with open(ORIGINAL_TEXT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    data = [item['context'] for item in data if 'input' in item]
    results = chunk_documents(
        data,
        max_token_size=max_token_size,
        overlap_token_size=overlap_token_size,
    )
    with open(CHUNK_TEXT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main(DATASET_ROOT, DATASET, MAX_TOKEN_SIZE, OVERLAP_TOKEN_SIZE)