import json
import re

# Extract json function
def extract_json(response: str) -> dict:
    try:
        return json.loads(response)
    except:
        json_pattern = r'(\{.*\})'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            json_str = match.group(1)
            json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
            json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1"\2":', json_str)
            return json.loads(json_str)
        return {}
