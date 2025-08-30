# Add this code snippet to show how to integrate monitoring into the training loop

def train_with_focal_loss_monitored(
    model,
    train_loader,
    val_loader=None,
    epochs=3,
    learning_rate=1e-5,
    device="cuda",
    tokenizer=None,
    save_checkpoint_at_batch=2000,
    checkpoint_path="checkpoint_batch_2000.pt",
    early_stopping_patience=10,
    early_stopping_metric='val_f1'  # or 'val_loss' or 'val_class_1_acc'
):
    """Training loop with monitoring, early stopping, and visualization"""
    
    # Initialize monitoring tools
    early_stopper = EarlyStopping(
        patience=early_stopping_patience, 
        min_delta=0.001,
        monitor=early_stopping_metric
    )
    visualizer = TrainingVisualizer()
    
    # ... (rest of your existing setup code) ...
    
    # Inside your training loop, after computing metrics:
    # (Add this after you calculate class accuracies)
    
    # Calculate F1 score for class 1
    if class_1_total > 0:
        true_positives = class_1_correct
        false_positives = pred_1_count - class_1_correct
        false_negatives = class_1_total - class_1_correct
        train_f1 = calculate_f1_score(true_positives, false_positives, false_negatives)
    else:
        train_f1 = 0.0
    
    # Update visualizer with training metrics
    train_class_0_acc = class_0_correct / class_0_total if class_0_total > 0 else 0
    train_class_1_acc = class_1_correct / class_1_total if class_1_total > 0 else 0
    pred_0_ratio = pred_0_count / (pred_0_count + pred_1_count) if (pred_0_count + pred_1_count) > 0 else 0
    pred_1_ratio = pred_1_count / (pred_0_count + pred_1_count) if (pred_0_count + pred_1_count) > 0 else 0
    
    visualizer.update(
        batch_idx,
        train_loss=loss.item() * gradient_accumulation_steps,
        train_class_0_acc=train_class_0_acc,
        train_class_1_acc=train_class_1_acc,
        train_f1=train_f1,
        pred_0_ratio=pred_0_ratio,
        pred_1_ratio=pred_1_ratio,
        val_loss=None,  # Will be updated during validation
        val_class_0_acc=None,
        val_class_1_acc=None,
        val_f1=None
    )
    
    # During validation (inside the validation check section):
    if val_loader is not None and batch_idx > 0 and batch_idx % validation_frequency == 0:
        print(f"\n--- Validation Check at Batch {batch_idx} ---")
        val_metrics = run_validation(model, val_loader, max_batches=100)
        
        # Calculate validation F1
        if val_metrics['class_1_total'] > 0:
            val_tp = int(val_metrics['class_1_acc'] * val_metrics['class_1_total'])
            val_fp = 0  # Would need to track this separately
            val_fn = val_metrics['class_1_total'] - val_tp
            val_f1 = calculate_f1_score(val_tp, val_fp, val_fn)
        else:
            val_f1 = 0.0
        
        # Update visualizer with validation metrics
        visualizer.metrics['val_loss'][-1] = val_metrics['loss']
        visualizer.metrics['val_class_0_acc'][-1] = val_metrics['class_0_acc']
        visualizer.metrics['val_class_1_acc'][-1] = val_metrics['class_1_acc']
        visualizer.metrics['val_f1'][-1] = val_f1
        
        print(f"Validation Loss: {val_metrics['loss']:.4f}")
        print(f"Validation Class 0 accuracy: {val_metrics['class_0_acc']:.2%}")
        if val_metrics['class_1_total'] > 0:
            print(f"Validation Class 1 accuracy: {val_metrics['class_1_acc']:.2%}")
            print(f"Validation F1 Score: {val_f1:.4f}")
        print("--- End Validation Check ---\n")
        
        # Check early stopping
        if early_stopping_metric == 'val_loss':
            early_stopper(val_metrics['loss'], model)
        elif early_stopping_metric == 'val_class_1_acc':
            early_stopper(val_metrics['class_1_acc'], model)
        elif early_stopping_metric == 'val_f1':
            early_stopper(val_f1, model)
            
        if early_stopper.early_stop:
            print("\n⚠️  EARLY STOPPING TRIGGERED!")
            print(f"Loading best model from {early_stopper.counter} validations ago")
            early_stopper.load_best_model(model)
            
            # Plot final results
            visualizer.plot(save_path=f'training_progress_epoch_{epoch+1}.png', show=False)
            visualizer.print_summary()
            
            return model
    
    # At the end of each epoch, plot progress
    visualizer.plot(save_path=f'training_progress_epoch_{epoch+1}.png', show=False)
    
    # After training completes
    visualizer.print_summary()
    return model


# Example usage snippet:
"""
# When calling the training function:
trained_model = train_with_focal_loss_monitored(
    model,
    train_loader,
    val_loader,
    epochs=3,
    learning_rate=1e-4,
    device=device,
    tokenizer=tokenizer,
    save_checkpoint_at_batch=2000,
    checkpoint_path="checkpoint_batch_2000.pt",
    early_stopping_patience=5,  # Stop if no improvement for 5 validations
    early_stopping_metric='val_f1'  # Monitor F1 score for minority class
)
"""