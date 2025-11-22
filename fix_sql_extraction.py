import json
import re

# Load the generated results
with open('output/llama3.1-8b-spider-100samples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def extract_sql_from_text(text):
    """Extract SQL query from LLM-generated text that may include explanation."""
    if not text or not isinstance(text, str):
        return "SELECT 1"
    
    # Remove leading "SELECT" if it's just a prefix before explanation
    text = text.strip()
    
    # Common patterns in llama output
    # Pattern 1: "SELECT To answer..." -> remove everything before actual SQL
    if text.startswith("SELECT To "):
        # Try to find actual SQL in code blocks
        code_blocks = re.findall(r'```sql\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
        if code_blocks:
            return code_blocks[0].strip()
        
        code_blocks = re.findall(r'```\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            sql = code_blocks[0].strip()
            if sql.upper().startswith('SELECT'):
                return sql
        
        # Try to find SELECT statement after explanation
        select_match = re.search(r'(SELECT\s+(?:DISTINCT\s+)?[\w\s,.*()\[\]]+FROM\s+[^;]+)', text, re.IGNORECASE | re.DOTALL)
        if select_match:
            return select_match.group(1).strip()
    
    # If it looks like valid SQL already, return it
    if text.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
        # Take everything up to semicolon or end
        sql = text.split(';')[0].strip()
        return sql
    
    # Last resort - return original or placeholder
    return text if len(text) < 500 else "SELECT 1"

# Process each result
fixed_count = 0
for item in data:
    if 'infer' in item:
        original = item['infer']
        extracted = extract_sql_from_text(original)
        if extracted != original:
            item['infer'] = extracted
            fixed_count += 1

# Save fixed results
with open('output/llama3.1-8b-spider-100samples.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✓ Fixed {fixed_count} SQL extractions out of {len(data)} total")
print(f"✓ Updated: output/llama3.1-8b-spider-100samples.json")
