import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_score, model):
        # For loss, lower is better. For accuracy, higher is better
        if 'loss' in self.monitor:
            score = -val_score
        else:
            score = val_score
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'Early stopping triggered! Best {self.monitor}: {-self.best_score if "loss" in self.monitor else self.best_score:.4f}')
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        """Save model state when validation score improves"""
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
    def load_best_model(self, model):
        """Load the best model state"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            

class TrainingVisualizer:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.batch_numbers = []
        
    def update(self, batch_num, **kwargs):
        self.batch_numbers.append(batch_num)
        for key, value in kwargs.items():
            self.metrics[key].append(value)
            
    def plot(self, save_path='training_progress.png', show=True):
        """Create a comprehensive training visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        if 'train_loss' in self.metrics:
            ax1.plot(self.batch_numbers, self.metrics['train_loss'], label='Train Loss', alpha=0.7)
        if 'val_loss' in self.metrics:
            val_batches = [b for b, v in zip(self.batch_numbers, self.metrics['val_loss']) if v is not None]
            val_losses = [v for v in self.metrics['val_loss'] if v is not None]
            ax1.plot(val_batches, val_losses, label='Val Loss', marker='o', markersize=6)
        ax1.set_xlabel('Batch Number')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for loss
        
        # Plot 2: Class accuracies
        ax2 = axes[0, 1]
        if 'train_class_0_acc' in self.metrics:
            ax2.plot(self.batch_numbers, self.metrics['train_class_0_acc'], 
                    label='Train Class 0 Acc', color='blue', alpha=0.7)
        if 'train_class_1_acc' in self.metrics:
            ax2.plot(self.batch_numbers, self.metrics['train_class_1_acc'], 
                    label='Train Class 1 Acc', color='red', alpha=0.7)
        if 'val_class_0_acc' in self.metrics:
            val_batches = [b for b, v in zip(self.batch_numbers, self.metrics['val_class_0_acc']) if v is not None]
            val_acc = [v for v in self.metrics['val_class_0_acc'] if v is not None]
            ax2.plot(val_batches, val_acc, label='Val Class 0 Acc', 
                    color='blue', marker='o', markersize=6, linestyle='--')
        if 'val_class_1_acc' in self.metrics:
            val_batches = [b for b, v in zip(self.batch_numbers, self.metrics['val_class_1_acc']) if v is not None]
            val_acc = [v for v in self.metrics['val_class_1_acc'] if v is not None]
            ax2.plot(val_batches, val_acc, label='Val Class 1 Acc', 
                    color='red', marker='o', markersize=6, linestyle='--')
        ax2.set_xlabel('Batch Number')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Class-wise Accuracies')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        # Plot 3: Prediction distribution
        ax3 = axes[1, 0]
        if 'pred_0_ratio' in self.metrics and 'pred_1_ratio' in self.metrics:
            ax3.plot(self.batch_numbers, self.metrics['pred_0_ratio'], 
                    label='Pred 0 Ratio', color='blue', alpha=0.7)
            ax3.plot(self.batch_numbers, self.metrics['pred_1_ratio'], 
                    label='Pred 1 Ratio', color='red', alpha=0.7)
            # Add true class distribution lines
            ax3.axhline(y=0.965, color='blue', linestyle=':', alpha=0.5, label='True 0 Ratio')
            ax3.axhline(y=0.035, color='red', linestyle=':', alpha=0.5, label='True 1 Ratio')
        ax3.set_xlabel('Batch Number')
        ax3.set_ylabel('Prediction Ratio')
        ax3.set_title('Prediction Distribution vs True Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)
        
        # Plot 4: Combined metric (F1 or balanced accuracy)
        ax4 = axes[1, 1]
        if 'train_f1' in self.metrics:
            ax4.plot(self.batch_numbers, self.metrics['train_f1'], 
                    label='Train F1', color='green', alpha=0.7)
        if 'val_f1' in self.metrics:
            val_batches = [b for b, v in zip(self.batch_numbers, self.metrics['val_f1']) if v is not None]
            val_f1 = [v for v in self.metrics['val_f1'] if v is not None]
            ax4.plot(val_batches, val_f1, label='Val F1', 
                    color='green', marker='o', markersize=6, linestyle='--')
        ax4.set_xlabel('Batch Number')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('F1 Score (Harmonic Mean of Precision/Recall)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
    def print_summary(self):
        """Print a summary of the best achieved metrics"""
        print("\n=== Training Summary ===")
        if 'val_loss' in self.metrics:
            best_val_loss = min([v for v in self.metrics['val_loss'] if v is not None])
            print(f"Best Validation Loss: {best_val_loss:.4f}")
        if 'val_class_1_acc' in self.metrics:
            best_val_acc = max([v for v in self.metrics['val_class_1_acc'] if v is not None])
            print(f"Best Validation Class 1 Accuracy: {best_val_acc:.2%}")
        if 'val_f1' in self.metrics:
            best_val_f1 = max([v for v in self.metrics['val_f1'] if v is not None])
            print(f"Best Validation F1 Score: {best_val_f1:.4f}")


def calculate_f1_score(true_positives, false_positives, false_negatives):
    """Calculate F1 score for binary classification"""
    if true_positives == 0:
        return 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)