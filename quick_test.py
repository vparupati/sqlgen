#!/usr/bin/env python3
"""
Quick test script for Gen-SQL with Ollama and Spider dataset
Tests with a small sample (5 examples) using llama3.1:8b
"""

import os
import sys

# Set environment variables for Ollama with llama3.1:8b
os.environ['LLM_HOST'] = 'localhost'
os.environ['LLM_PORT'] = '11434'
os.environ['LLM_MODEL'] = 'llama3.1:8b'

# Import main.py components
import json
from main import LLMClient, Retriever, process_record_retrieve_tables_based_on_question, MODE, FEW_SHOT_COMPLETION
import time

def test_quick_spider():
    """Run a quick test with 5 Spider examples"""
    
    print("="*60)
    print("Gen-SQL Quick Test - Spider Dataset")
    print("="*60)
    print(f"Model: llama3.1:8b")
    print(f"Dataset: Spider (5 sample queries)")
    print("="*60)
    
    # Initialize LLM client
    print("\n1. Connecting to Ollama...")
    try:
        client = LLMClient(host='localhost', port=11434, model_name='llama3.1:8b')
    except Exception as e:
        print(f"\n❌ Failed to connect to Ollama!")
        print(f"Error: {e}")
        print("\nPlease ensure Ollama is running:")
        print("  ollama serve")
        sys.exit(1)
    
    # Initialize retriever
    print("\n2. Loading Spider database schemas...")
    retriever = Retriever(
        db_set_path='benchmark/spider/database/*',
        training_set_paths=['benchmark/spider/train_others.json', 'benchmark/spider/train_spider.json']
    )
    
    # Load dev set
    print("\n3. Loading test examples...")
    with open('benchmark/spider/dev.json', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take first 5 examples
    test_data = data[:5]
    
    # Add IDs
    for i, r in enumerate(test_data):
        r['id'] = i
    
    print(f"   Loaded {len(test_data)} test examples")
    
    # Process examples
    print("\n4. Running SQL generation...")
    print("-"*60)
    
    results = []
    start_time = time.time()
    
    for i, record in enumerate(test_data, 1):
        print(f"\nExample {i}/{len(test_data)}:")
        print(f"  Database: {record['db_id']}")
        print(f"  Question: {record['question'][:70]}...")
        
        try:
            result = process_record_retrieve_tables_based_on_question(
                client, retriever, record, FEW_SHOT_COMPLETION
            )
            
            if result:
                print(f"  ✓ Generated SQL:")
                print(f"    {result['infer'][:100]}...")
                results.append(result)
            else:
                print(f"  ✗ Failed to generate SQL")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}...")
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total examples: {len(test_data)}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(test_data) - len(results)}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Avg time per query: {elapsed/len(test_data):.2f} seconds")
    
    # Save results
    output_file = 'output/quick_test_spider_llama3.1-8b.json'
    os.makedirs('output', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("="*60)
    
    return results

if __name__ == '__main__':
    test_quick_spider()
