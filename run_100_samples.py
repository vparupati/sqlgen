#!/usr/bin/env python3
"""
Run Gen-SQL on 100 Spider samples and prepare for evaluation
"""

import os
import sys
os.environ['LLM_HOST'] = 'localhost'
os.environ['LLM_PORT'] = '11434'
os.environ['LLM_MODEL'] = 'llama3.1:8b'

import json
from main import LLMClient, Retriever, process_record_retrieve_tables_based_on_question, FEW_SHOT_COMPLETION
from tqdm import tqdm
import time

def run_100_samples():
    print("="*60)
    print("Gen-SQL Evaluation - 100 Spider Samples")
    print("="*60)
    print(f"Model: llama3.1:8b (Ollama)")
    print(f"Dataset: Spider dev set (100 samples)")
    print("="*60)
    
    # Initialize
    print("\n1. Connecting to Ollama...")
    client = LLMClient(host='localhost', port=11434, model_name='llama3.1:8b')
    
    print("\n2. Loading retriever...")
    retriever = Retriever(
        db_set_path='benchmark/spider/database/*',
        training_set_paths=['benchmark/spider/train_others.json', 'benchmark/spider/train_spider.json']
    )
    
    # Load dev set
    print("\n3. Loading dev set...")
    with open('benchmark/spider/dev.json', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take first 100
    test_data = data[:100]
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
                client, retriever, record, FEW_SHOT_COMPLETION
            )
            if result:
                results.append(result)
            else:
                # Add placeholder for failed examples
                results.append({
                    'id': record['id'],
                    'question': record['question'],
                    'output': record['query'],
                    'db': record['db_id'],
                    'infer': 'SELECT 1',  # Placeholder
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
                'infer': 'SELECT 1',  # Placeholder
                'usage': {},
                'time': 0
            })
    
    elapsed = time.time() - start_time
    
    # Save results
    output_file = 'output/llama3.1-8b-spider-100samples.json'
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
    
    return output_file

if __name__ == '__main__':
    output_file = run_100_samples()
    print("\nNext steps:")
    print(f"1. Convert output: python spider_code/convert_output_100.py")
    print(f"2. Evaluate: bash spider_code/eval-sql.sh")
