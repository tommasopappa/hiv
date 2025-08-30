"""
Complete implementation of Pythia-160m with custom PE using HuggingFace PEFT
Ready to run in Google Colab with your existing dataloader
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional

# Your PE module
class HIVPEModule(nn.Module):
    def __init__(self, pe_dim=30, embed_dim=768):
        super().__init__()
        self.pe_projection = nn.Sequential(
            nn.Linear(pe_dim, embed_dim),
            nn.GELU()
        )
        print(f"✓ PE Module initialized: {pe_dim} -> {embed_dim}")
    
    def forward(self, embeddings, graph_pes):
        if graph_pes is not None:
            projected_pes = self.pe_projection(graph_pes)
            return embeddings + projected_pes
        return embeddings

class PythiaWithPE(nn.Module):
    """Wrapper that adds PE to Pythia model"""
    def __init__(self, base_model, pe_dim=30):
        super().__init__()
        self.base_model = base_model
        self.embed_dim = base_model.gpt_neox.embed_in.embedding_dim
        self.pe_module = HIVPEModule(pe_dim=pe_dim, embed_dim=self.embed_dim)
        
    def forward(
        self,
        input_ids=None,
        graph_pes=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # Get embeddings
        inputs_embeds = self.base_model.gpt_neox.embed_in(input_ids)
        
        # Add PE
        inputs_embeds = self.pe_module(inputs_embeds, graph_pes)
        
        # Forward through the model with embeddings
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def generate(self, input_ids, graph_pes=None, **kwargs):
        """Generate with PE support"""
        # Get embeddings with PE for the prompt
        inputs_embeds = self.base_model.gpt_neox.embed_in(input_ids)
        inputs_embeds = self.pe_module(inputs_embeds, graph_pes)
        
        # Generate using embeddings
        return self.base_model.generate(
            inputs_embeds=inputs_embeds,
            **kwargs
        )

def setup_model_with_pe_and_lora(model_name="EleutherAI/pythia-160m", pe_dim=30):
    """
    Setup Pythia with custom PE and LoRA using PEFT
    
    Returns:
        model: Model with PE module and LoRA applied
        tokenizer: Tokenizer for the model
    """
    print(f"Loading {model_name}...")
    
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Wrap with PE
    model = PythiaWithPE(base_model, pe_dim=pe_dim)
    
    # Configure LoRA for Pythia
    # Target the attention and MLP modules in Pythia
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "query_key_value",  # Attention QKV
            "dense",            # Attention output
            "dense_h_to_4h",    # MLP up
            "dense_4h_to_h"     # MLP down
        ],
    )
    
    # Apply LoRA to the base model
    model.base_model = get_peft_model(model.base_model, lora_config)
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer

def train_with_pe(
    model,
    train_loader,
    val_loader=None,
    epochs=3,
    learning_rate=1e-4,
    device="cuda"
):
    """
    Training loop that handles graph PEs
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            graph_pes = batch['graph_pes'].to(device).half()
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                graph_pes=graph_pes,
                labels=labels
            )
            class_weights = torch.tensor([1.0, 33.5]).to(device)  # 971/29 ≈ 33.5

            # In your training loop, modify the loss calculation:
            outputs = model(input_ids=input_ids, graph_pes=graph_pes, labels=labels)

            # Custom weighted loss (since it's token-level)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    graph_pes = batch['graph_pes'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        graph_pes=graph_pes,
                        labels=labels
                    )
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
    
    return model

# Inference function
def generate_with_pe(model, tokenizer, prompt, graph_pe, max_length=100, device="cuda"):
    """
    Generate text with custom PE
    
    Args:
        model: The model with PE
        tokenizer: Tokenizer
        prompt: Text prompt
        graph_pe: Graph PE tensor of shape (seq_len, pe_dim)
        max_length: Maximum generation length
    """
    model.eval()
    model = model.to(device)
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    
    # Prepare PE
    if graph_pe.dim() == 2:
        graph_pe = graph_pe.unsqueeze(0)  # Add batch dimension
    graph_pe = graph_pe.to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            graph_pes=graph_pe,
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Complete Colab script
def colab_complete_setup():
    """
    Complete setup for Google Colab
    Returns everything needed for training
    """
    # Install dependencies
    import subprocess
    subprocess.run(["pip", "install", "transformers", "peft", "torch", "-q"])
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU found, using CPU")
    
    # Setup model
    model, tokenizer = setup_model_with_pe_and_lora()
    
    return model, tokenizer, device

# Example usage
if __name__ == "__main__":
    print("Pythia-160m with Custom PE using PEFT")
    print("=" * 50)
    
    # Quick test
    model, tokenizer = setup_model_with_pe_and_lora()
    
    # Test forward pass
    test_input = torch.randint(0, 50000, (2, 128))
    test_pe = torch.randn(2, 128, 30)
    
    with torch.no_grad():
        outputs = model(input_ids=test_input, graph_pes=test_pe)
        print(f"✓ Forward pass successful! Output shape: {outputs.logits.shape}")
    
    print("\nModel is ready for training with your dataloader!")