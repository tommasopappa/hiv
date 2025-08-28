import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

# Import your graph PE functions
from graph_pe_utils import compute_graph_pe, batch_compute_graph_pe_dict, align_graph_pe_to_tokens


class HIVDatasetWithGraphPE(Dataset):
    """
    Dataset for HIV classification with graph positional encodings.
    """
    def __init__(self, jsonl_file: str, tokenizer, max_length: int = 256, 
                 pe_dim: int = 30, precomputed_pe_file: Optional[str] = None):
        """
        Args:
            jsonl_file: Path to JSONL file with HIV data
            tokenizer: LitGPT tokenizer instance
            max_length: Maximum sequence length
            pe_dim: Dimension of positional encodings (number of eigenvectors)
            precomputed_pe_file: Optional path to precomputed graph PEs
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pe_dim = pe_dim
        self.data = []
        
        # Load data from JSONL
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Load or compute graph PEs
        if precomputed_pe_file and Path(precomputed_pe_file).exists():
            with open(precomputed_pe_file, 'rb') as f:
                self.precomputed_pes = pickle.load(f)
        else:
            self.precomputed_pes = None
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract SMILES from input
        smiles = item['input'].replace("SMILES: ", "")
        
        # Combine instruction and input for the prompt
        text = f"{item['instruction']} {item['input']} Answer:"
        
        # Tokenize using LitGPT tokenizer
        input_ids = self.tokenizer.encode(text, max_length=self.max_length)
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            input_ids = input_ids + [self.tokenizer.eos_id] * (self.max_length - len(input_ids))
        
        input_ids = torch.tensor(input_ids[:self.max_length])
        
        # Get or compute graph PE
        if self.precomputed_pes and smiles in self.precomputed_pes:
            graph_pe = self.precomputed_pes[smiles]
        else:
            # Compute graph PE on the fly
            graph_pe = compute_graph_pe(smiles, k=self.pe_dim)
        
        # Align graph PE to tokens
        if graph_pe is not None:
            aligned_pe = align_graph_pe_to_tokens(
                smiles=smiles,
                graph_pe=graph_pe,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
        else:
            # Use zero PE if graph computation fails
            aligned_pe = torch.zeros(self.max_length, self.pe_dim)
        
        # Create label (for next token prediction, shift by 1)
        # For classification, we'll use a special approach in the training loop
        label = int(item['output'])
        
        return {
            'input_ids': input_ids,
            'graph_pes': aligned_pe,
            'label': label,  # 0 or 1 for HIV activity
            'text': text,
            'expected_output': item['output']
        }


def create_dataloaders_with_pe(train_file: str, val_file: str, tokenizer, 
                              batch_size: int = 8, max_length: int = 256,
                              pe_dim: int = 30, num_workers: int = 4):
    """
    Create train and validation dataloaders with graph PE support.
    """
    train_dataset = HIVDatasetWithGraphPE(
        train_file, tokenizer, max_length, pe_dim
    )
    
    val_dataset = HIVDatasetWithGraphPE(
        val_file, tokenizer, max_length, pe_dim
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def precompute_graph_pes(jsonl_file: str, pe_dim: int = 30, 
                        output_file: str = "precomputed_pes.pkl"):
    """
    Precompute graph PEs for all molecules in the dataset to speed up training.
    """
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Extract all SMILES
    smiles_list = [item['input'].replace("SMILES: ", "") for item in data]
    
    # Compute all PEs at once using batch function
    pes = batch_compute_graph_pe_dict(smiles_list, k=pe_dim)
    
    # Save precomputed PEs
    with open(output_file, 'wb') as f:
        pickle.dump(pes, f)
    
    # Count non-None PEs
    valid_pes = sum(1 for pe in pes.values() if pe is not None)
    print(f"Saved {valid_pes} valid PEs out of {len(pes)} total to {output_file}")
    
    return pes
