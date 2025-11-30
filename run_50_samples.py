#!/usr/bin/env python3
"""
Run Gen-SQL on 50 Spider samples for evaluation
"""

import os
import sys
# os.environ['LLM_HOST'] = 'localhost'
# os.environ['LLM_PORT'] = '11434'
# os.environ['LLM_MODEL'] = 'llama3.1:8b'

import json
from main import LLMClient, Retriever, process_record_retrieve_tables_based_on_question, FEW_SHOT_COMPLETION, ZERO_SHOT_CHAT
import main
main.ENABLE_SQL_POST_PROCESS = False
from tqdm import tqdm
import time

def run_50_samples():
    print("="*60)
    print("Gen-SQL Evaluation - 50 Spider Samples")
    print("="*60)
    print(f"Model: llama3.1:8b (Ollama)")
    print(f"Dataset: Spider dev set (50 samples)")
    print("="*60)
    
    # Initialize
    print("\n1. Connecting to Ollama...")
    llm_host = os.getenv('LLM_HOST', 'localhost')
    llm_port = int(os.getenv('LLM_PORT', '11434'))
    llm_model = os.getenv('LLM_MODEL', 'llama3.1:8b')
    client = LLMClient(host=llm_host, port=llm_port, model_name=llm_model)
    
    print("\n2. Loading retriever...")
    retriever = Retriever(
        db_set_path='benchmark/spider/database/*',
        training_set_paths=['benchmark/spider/train_others.json', 'benchmark/spider/train_spider.json']
    )
    
    # Load dev set
    print("\n3. Loading dev set...")
    with open('benchmark/spider/dev.json', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take first 50
    test_data = data[:50]
    for i, r in enumerate(test_data):
        r['id'] = i
    
    print(f"   Loaded {len(test_data)} examples")
    
    # Process
    print(f"\n4. Processing {len(test_data)} examples...")
    results = []
    start_time = time.time()
    
    for record in tqdm(test_data, desc="Generating SQL"):
        try:
            result = process_record_retrieve_tables_based_on_question(
                client, retriever, record, ZERO_SHOT_CHAT
            )
            if result:
                results.append(result)
            else:
                # Placeholder for failure
                results.append({
                    'id': record['id'],
                    'question': record['question'],
                    'output': record['query'],
                    'db': record['db_id'],
                    'infer': 'SELECT 1',
                    'usage': {},
                    'time': 0
                })
        except Exception as e:
            print(f"\nError on example {record['id']}: {str(e)[:100]}")
            results.append({
                'id': record['id'],
                'question': record['question'],
                'output': record['query'],
                'db': record['db_id'],
                'infer': 'SELECT 1',
                'usage': {},
                'time': 0
            })
    
    elapsed = time.time() - start_time
    
    # Save results
    output_file = 'output/llama3.1-8b-spider-50samples.json'
    os.makedirs('output', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total examples: {len(test_data)}")
    print(f"Completed: {len(results)}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Avg time: {elapsed/len(test_data):.2f} sec/query")
    print(f"\n✓ Results saved to: {output_file}")
    print("="*60)

if __name__ == '__main__':
    run_50_samples()
