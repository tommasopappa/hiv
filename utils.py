import json
from transformers import AutoTokenizer
import numpy as np
def load_jsonl(filepath):
    """Load data from JSONL file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def calculate_max_lengths(data_path):
    """Calculate max lengths for SMILES and full prompts"""

    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    data = load_jsonl(data_path)
    
    smiles_lengths = []
    full_prompt_lengths = []
    
    for i,item in enumerate(data):
        
        # Extract SMILES
        smiles = item['input'].replace("SMILES: ", "")
        
        # Tokenize just SMILES
        smiles_tokens = tokenizer(smiles, add_special_tokens=False)
        token_ids = smiles_tokens['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(token_ids) 
        if i == 0:
            smilestest=smiles
            tokenstest=tokens
            
        smiles_lengths.append(len(smiles_tokens['input_ids']))
        
        # Tokenize full prompt + output
        full_text = f"{item['instruction']}\n{item['input']}\nAnswer: {item['output']}"
        full_tokens = tokenizer(full_text, add_special_tokens=True)
        full_prompt_lengths.append(len(full_tokens['input_ids']))
   
    
  
    recommended_smiles = int(np.max(smiles_lengths))
    recommended_full = int(np.max(full_prompt_lengths))
    
    #print(f"\nRecommended max_smiles_length: {recommended_smiles}")
    #print(f"Recommended max_length: {recommended_full}")
    
    return recommended_smiles, recommended_full

