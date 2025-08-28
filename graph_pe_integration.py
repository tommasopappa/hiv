import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from typing import Optional, Dict, Any
from litgpt.model import GPT
from litgpt.config import Config

class EmbeddingWithGraphPE(nn.Module):
    def __init__(self, embed_dim=512, pe_dim=30):
        super().__init__()
        # One-layer projection with GeLU from graph PE to embedding dimension
        # Using standard Laplacian, so pe_dim is just k
        self.pe_projection = nn.Sequential(
            nn.Linear(pe_dim, embed_dim),
            nn.GELU()
        )

    def forward(self, embeddings, graph_pes):
        # token_pes shape: [batch_size, seq_len, pe_dim]
        # embeddings shape: [batch_size, seq_len, embed_dim]

        # Project token PEs to embedding dimension with GeLU
        projected_pes = self.pe_projection(graph_pes)

        # Add to token embeddings
        enhanced_embeddings = embeddings + projected_pes

        return enhanced_embeddings


class GPTWithGraphPE(GPT):
    """Extended GPT model that incorporates graph positional encodings."""
    
    def __init__(self, config: Config, pe_dim: int = 30):
        super().__init__(config)
        
        # Add the graph PE module
        self.graph_pe_module = EmbeddingWithGraphPE(
            embed_dim=config.n_embd,
            pe_dim=pe_dim
        )
        
    def forward(self, idx: torch.Tensor, graph_pes: Optional[torch.Tensor] = None, 
                input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional graph positional encodings.
        
        Args:
            idx: Input token indices [batch_size, seq_len]
            graph_pes: Graph positional encodings [batch_size, seq_len, pe_dim]
            input_pos: Position indices (for KV cache)
        """
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Sequence length {T} exceeds model's max_seq_length {self.max_seq_length}")
        
        # Get token embeddings
        x = self.transformer.wte(idx)  # [batch_size, seq_len, n_embd]
        
        # Apply graph PE if provided
        if graph_pes is not None:
            x = self.graph_pe_module(x, graph_pes)
        
        # Continue with standard GPT forward pass
        if input_pos is not None:
            rope = self.rope.forward(idx, input_pos)
            mask = self.mask.forward(idx, input_pos)
        else:
            rope = self.rope.forward(idx)
            mask = self.mask.forward(idx)
            
        for block in self.transformer.h:
            x = block(x, rope, mask, input_pos)
            
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        
        return x


def setup_model_with_graph_pe(checkpoint_path: str, pe_dim: int = 30) -> GPTWithGraphPE:
    """
    Load a pretrained model and add graph PE support.
    
    Args:
        checkpoint_path: Path to the pretrained model checkpoint
        pe_dim: Dimension of graph positional encodings
    """
    # Load the config
    config = Config.from_file(Path(checkpoint_path) / "model_config.yaml")
    
    # Create model with graph PE
    model = GPTWithGraphPE(config, pe_dim=pe_dim)
    
    # Load pretrained weights (excluding graph_pe_module which will be trained)
    checkpoint = torch.load(Path(checkpoint_path) / "lit_model.pth", map_location="cpu")
    
    # Filter out any graph_pe_module weights if they exist
    state_dict = {k: v for k, v in checkpoint.items() if not k.startswith("graph_pe_module")}
    
    # Load weights with strict=False to allow missing graph_pe_module weights
    model.load_state_dict(state_dict, strict=False)
    
    return model


def apply_lora_with_graph_pe(model: GPTWithGraphPE, rank: int = 8, alpha: int = 16, 
                            dropout: float = 0.0, target_modules: Optional[list] = None):
    """
    Apply LoRA to the model while keeping graph PE module trainable.
    
    Args:
        model: GPTWithGraphPE model
        rank: LoRA rank
        alpha: LoRA alpha scaling parameter
        dropout: LoRA dropout
        target_modules: List of module names to apply LoRA to
    """
    from litgpt.lora import GPT as LoRAGPT, Config as LoRAConfig
    
    # Default target modules for LoRA
    if target_modules is None:
        target_modules = ["wq", "wk", "wv", "wo", "wi", "w2"]
    
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze graph PE module parameters
    for param in model.graph_pe_module.parameters():
        param.requires_grad = True
    
    # Apply LoRA to target modules
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            # Apply LoRA logic here (simplified - you'd use actual LoRA implementation)
            # This is where you'd integrate with LitGPT's LoRA implementation
            pass
    
    return model
