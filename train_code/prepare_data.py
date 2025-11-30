import json
import os
import sys
import random
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sql_utils_2 import dump_db_json_schema, build_sql

def format_schema(db_id, db_path):
    db_file = os.path.join(db_path, db_id, f"{db_id}.sqlite")
    try:
        tables = dump_db_json_schema(db_file)
    except Exception as e:
        print(f"Error loading schema for {db_id}: {e}")
        return ""
    
    schema_strs = []
    for table in tables:
        # We don't have descriptions in the training data easily accessible without loading another file
        # For simplicity, we skip descriptions for now or load them if needed
        # build_sql(table_name, columns, primary_keys, foreign_keys, descriptions=None)
        table_sql = build_sql(
            table['name'],
            table['columns'],
            table['primary_keys'],
            table['foreign_keys'],
            descriptions={}
        )
        schema_strs.append(table_sql)
    
    return "\n".join(schema_strs)

def format_llama3_chat(schema, question, sql):
    # Llama 3 format
    # <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    # 
    # {content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    # 
    # {content}<|eot_id|>
    
    user_content = f"Write a SQL query to answer the question. Output ONLY the SQL query without any explanations or text.\nDatabase Schema:\n{schema}\n\nQuestion: {question}"
    
    text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{sql}<|eot_id|>"
    return text

def main():
    print("Preparing data for Llama 3 fine-tuning...")
    
    spider_dir = "benchmark/spider"
    train_files = ["train_spider.json", "train_others.json"]
    db_path = os.path.join(spider_dir, "database")
    
    all_data = []
    
    # Cache schemas to speed up
    schema_cache = {}
    
    for train_file in train_files:
        fpath = os.path.join(spider_dir, train_file)
        print(f"Loading {fpath}...")
        with open(fpath, 'r') as f:
            data = json.load(f)
            
        for item in tqdm(data, desc=f"Processing {train_file}"):
            db_id = item['db_id']
            question = item['question']
            sql = item['query']
            
            if db_id not in schema_cache:
                schema_cache[db_id] = format_schema(db_id, db_path)
            
            schema = schema_cache[db_id]
            if not schema:
                continue
                
            text = format_llama3_chat(schema, question, sql)
            all_data.append({"text": text})
    
    print(f"Total examples: {len(all_data)}")
    
    # Split train/valid
    random.seed(42)
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    valid_data = all_data[split_idx:]
    
    print(f"Train size: {len(train_data)}")
    print(f"Valid size: {len(valid_data)}")
    
    # Save
    os.makedirs("data", exist_ok=True)
    
    with open("data/train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open("data/valid.jsonl", "w") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
            
    print("Data saved to data/train.jsonl and data/valid.jsonl")

if __name__ == "__main__":
    main()
