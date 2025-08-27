import itertools
import torch
from torch.utils.data import IterableDataset
import pandas as pd
import os

class PartialHIVDataset(IterableDataset):
    """
    IterableDataset that loads only a small portion of the HIV dataset.
    """
    
    def __init__(self, root='/tmp/HIV', max_samples=10):
        """
        Args:
            root: Root directory for dataset storage
            max_samples: Maximum number of samples to load
        """
        self.root = root
        self.max_samples = max_samples
        self._dataset = None
    
    def _lazy_load_dataset(self):
        """Lazy load the dataset only when iteration begins"""
        if self._dataset is None:
            print(f"Initializing HIV dataset (will only load {self.max_samples} samples)...")
            # Import here to ensure it's available
            from torch_geometric.datasets import MoleculeNet
            self._dataset = MoleculeNet(root=self.root, name='HIV')
            print(f"Dataset ready! Total size: {len(self._dataset)} molecules")
            print(f"But we'll only load {self.max_samples} of them.\n")
    
    def parse_molecules(self):
        """
        Parse molecules from the dataset, stopping after max_samples.
        """
        self._lazy_load_dataset()
        
        for i in range(min(self.max_samples, len(self._dataset))):
            data = self._dataset[i]
            
            # Extract molecule information
            mol_info = {
                'source': f"molecule_{i}",  # Similar to your CustomIterableDataset
                'target': data.y.item(),    # HIV activity label
                'num_atoms': data.num_nodes,
                'num_bonds': data.num_edges // 2,  # Undirected edges
                'smiles': getattr(data, 'smiles', 'N/A')
            }
            
            yield mol_info
    
    def __iter__(self):
        """Iterator with worker support"""
        iterator = self.parse_molecules()
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id
            return itertools.islice(iterator, worker_id, None, worker_total_num)
        
        return iterator
