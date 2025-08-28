import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, Optional
import wandb
from transformers import AutoTokenizer

from litgpt import Tokenizer
from litgpt.lora import mark_only_lora_as_trainable
from litgpt.utils import num_parameters

from graph_pe_integration import GPTWithGraphPE, setup_model_with_graph_pe
from dataset_litgpt import create_dataloaders_with_pe, precompute_graph_pes


def train_with_graph_pe(
    model_name: str = "EleutherAI/pythia-160m",  # Small Pythia model
    train_file: str = "hiv_train.jsonl",
    val_file: str = "hiv_val.jsonl",
    output_dir: str = "checkpoints/hiv_graph_pe",
    # Training hyperparameters
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    num_epochs: int = 3,
    warmup_steps: int = 100,
    max_length: int = 256,  # Shorter for small models
    # LoRA hyperparameters
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # Graph PE hyperparameters
    pe_dim: int = 30,
    # Other settings
    gradient_accumulation_steps: int = 4,
    eval_steps: int = 50,
    save_steps: int = 200,
    logging_steps: int = 10,
    use_wandb: bool = False,
):
    """
    Fine-tune a LitGPT model with LoRA and graph positional encodings for HIV classification.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project="hiv-classification-graph-pe",
            config={
                "model_name": model_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "pe_dim": pe_dim,
            }
        )
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Initialize ChemBERTa tokenizer
    print("Loading ChemBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    vocab_size = len(tokenizer)
    print(f"Tokenizer vocabulary size: {vocab_size}")
    
    # Step 2: Precompute graph PEs (optional but recommended)
    print("Precomputing graph PEs...")
    pe_file_train = output_dir / "train_pes.pkl"
    pe_file_val = output_dir / "val_pes.pkl"
    
    if not pe_file_train.exists():
        precompute_graph_pes(train_file, pe_dim, pe_file_train)
    if not pe_file_val.exists():
        precompute_graph_pes(val_file, pe_dim, pe_file_val)
    
    # Step 3: Create data loaders with graph PE support
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders_with_pe(
        train_file=train_file,
        val_file=val_file,
        batch_size=batch_size,
        max_length=max_length,
        pe_dim=pe_dim,
    )
    
    # Step 4: Load model with graph PE support
    print("Loading model with graph PE support...")
    model = setup_model_with_graph_pe(
        checkpoint_path=f"checkpoints/{model_name}",
        pe_dim=pe_dim
    ).to(device)
    
    # Step 5: Apply LoRA
    print("Applying LoRA...")
    # This is simplified - you'd use LitGPT's actual LoRA implementation
    from litgpt.lora import lora_filter, mark_only_lora_as_trainable
    
    # Configure which layers get LoRA
    model = lora_filter(model, lora_r=lora_r, lora_alpha=lora_alpha, 
                       lora_dropout=lora_dropout)
    
    # Mark only LoRA and graph PE parameters as trainable
    mark_only_lora_as_trainable(model)
    
    # Additionally ensure graph PE module is trainable
    for param in model.graph_pe_module.parameters():
        param.requires_grad = True
    
    # Print parameter count
    total_params = num_parameters(model, requires_grad=False)
    trainable_params = num_parameters(model, requires_grad=True)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Step 6: Setup training
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    criterion = nn.CrossEntropyLoss()
    
    # Step 7: Training loop
    print("Starting training...")
    global_step = 0
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            graph_pes = batch['graph_pes'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with graph PEs
            outputs = model(input_ids, graph_pes=graph_pes)
            
            # Get logits for the last token (classification token)
            logits = outputs[:, -1, :]  # [batch_size, vocab_size]
            
            # For classification, we typically use specific tokens (e.g., "0" and "1")
            # Get token IDs for "0" and "1"
            token_0_id = tokenizer.encode("0", bos=False, eos=False)[0]
            token_1_id = tokenizer.encode("1", bos=False, eos=False)[0]
            
            # Extract logits for our classification tokens
            class_logits = torch.stack([
                logits[:, token_0_id],
                logits[:, token_1_id]
            ], dim=1)  # [batch_size, 2]
            
            loss = criterion(class_logits, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            train_loss += loss.item() * gradient_accumulation_steps
            
            # Calculate accuracy
            preds = class_logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = train_loss / (step + 1)
                    avg_acc = train_correct / train_total
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'acc': f'{avg_acc:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                    })
                    
                    if use_wandb:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/accuracy': avg_acc,
                            'train/lr': scheduler.get_last_lr()[0],
                            'step': global_step
                        })
                
                # Evaluation
                if global_step % eval_steps == 0:
                    val_acc = evaluate(model, val_loader, device, tokenizer)
                    
                    if use_wandb:
                        wandb.log({
                            'val/accuracy': val_acc,
                            'step': global_step
                        })
                    
                    # Save best model
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        save_checkpoint(model, optimizer, scheduler, global_step, 
                                      output_dir / "best_model")
                    
                    model.train()
                
                # Regular checkpointing
                if global_step % save_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step,
                                  output_dir / f"checkpoint-{global_step}")
        
        # End of epoch evaluation
        print(f"\nEpoch {epoch+1} completed")
        val_acc = evaluate(model, val_loader, device, tokenizer)
        print(f"Validation accuracy: {val_acc:.4f}")
    
    # Save final model
    save_checkpoint(model, optimizer, scheduler, global_step, output_dir / "final_model")
    
    if use_wandb:
        wandb.finish()
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")


def evaluate(model, dataloader, device, tokenizer):
    """Evaluate the model on the validation set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            graph_pes = batch['graph_pes'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, graph_pes=graph_pes)
            logits = outputs[:, -1, :]
            
            # Get classification logits
            token_0_id = tokenizer.encode("0", bos=False, eos=False)[0]
            token_1_id = tokenizer.encode("1", bos=False, eos=False)[0]
            
            class_logits = torch.stack([
                logits[:, token_0_id],
                logits[:, token_1_id]
            ], dim=1)
            
            preds = class_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return accuracy


def save_checkpoint(model, optimizer, scheduler, step, output_dir):
    """Save model checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
    }
    
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    
    # Save config
    config = {
        'model_class': 'GPTWithGraphPE',
        'pe_dim': model.graph_pe_module.pe_projection[0].in_features,
        'embed_dim': model.graph_pe_module.pe_projection[0].out_features,
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    # Example usage with small models
    # Option 1: Pythia-160m (very small, good for testing)
    # Option 2: Pythia-410m (small but more capable)
    # Option 3: Pythia-1b (larger but still efficient)
    
    train_with_graph_pe(
        model_name="EleutherAI/pythia-160m",  # 160M parameters
        # model_name="EleutherAI/pythia-410m",  # 410M parameters
        # model_name="EleutherAI/pythia-1b",  # 1B parameters
        train_file="hiv_train.jsonl",
        val_file="hiv_val.jsonl",
        batch_size=8,
        learning_rate=3e-4,
        num_epochs=3,
        pe_dim=30,
        use_wandb=False
    )
