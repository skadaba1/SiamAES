import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from loss import CrossEntropyLoss, ContrastiveLoss

class BaselineTrainer:
    def __init__(self, model, train_dataloader, eval_dataloader, device, lr=5e-5, weight_decay=0.01, num_epochs=3):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.num_training_steps = num_epochs * len(train_dataloader)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps
        )
        self.best_val_loss = float("inf")
        self.train_loss_history = []
        self.val_loss_history = []

    def _train_epoch(self):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0

        for batch in self.train_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    def _validate_epoch(self):
        """Evaluates the model on the validation set."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.eval_dataloader)
        return avg_loss

    def train(self):
        """Main training loop."""
        progress_bar = tqdm(range(self.num_training_steps), desc="Training")
        
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            print(f"Epoch {epoch+1}/{self.num_epochs} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model_checkpoint(epoch)
                print("Best model updated!")

            progress_bar.update(len(self.train_dataloader))

        print("Training complete.")
        return self.train_loss_history, self.val_loss_history

    def _save_model_checkpoint(self, epoch):
        """Saves the model checkpoint."""
        checkpoint_path = f"best_model_epoch_{epoch+1}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

class SiameseTrainer:
    def __init__(self, model, train_dataloader, eval_dataloader, device, num_epochs, num_classes, lr=5e-5, alpha=1, beta=1, gamma=1):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_func = ContrastiveLoss()
        self.loss_func_classification = CrossEntropyLoss()
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.num_training_steps = num_epochs * len(train_dataloader)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps
        )
        self.best_val_loss = float("inf")
        self.train_loss_history = []
        self.val_loss_history = []

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for sample1, sample2 in self.train_dataloader:
            sample1 = {k: v.to(self.device) for k, v in sample1.items()}
            sample2 = {k: v.to(self.device) for k, v in sample2.items()}
            
            out_x1, out_x2, update_x1, update_x2 = self.model(sample1, sample2, None)
            self.optimizer.zero_grad()

            flag = (sample1['labels'] == sample2['labels']).cpu().item()
            
            # Compute losses
            loss_val = self.gamma * self.loss_func(out_x1, out_x2, flag)
            loss_val.backward(retain_graph=True)
            self.optimizer.step()

            x1_classification_val = self.alpha * self.loss_func_classification(update_x1)
            x1_classification_val.backward(retain_graph=True)
            self.optimizer.step()

            x2_classification_val = self.beta * self.loss_func_classification(update_x2)
            x2_classification_val.backward()
            self.optimizer.step()

            total_loss += self.alpha * loss_val.item() + self.beta * x1_classification_val.item() + self.gamma * x2_classification_val.item()
            self.lr_scheduler.step()
        
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for sample1 in self.eval_dataloader:
                sample1 = {k: v.to(self.device) for k, v in sample1.items()}
                out_x1, update_x1 = self.model(sample1, None, None)
                x1_classification_val = self.loss_func_classification(update_x1)
                total_loss += x1_classification_val.detach().cpu().item()
        
        avg_loss = total_loss / len(self.eval_dataloader)
        return avg_loss

    def train(self):
        progress_bar = tqdm(range(self.num_training_steps), desc="Training")

        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            print(f"Epoch {epoch+1}/{self.num_epochs} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model_checkpoint(epoch)
                print("Saving checkpoint!")
            
            progress_bar.update(len(self.train_dataloader))

        print("Training complete.")
        return self.train_loss_history, self.val_loss_history

    def _save_model_checkpoint(self, epoch):
        checkpoint_path = f"best_siamese_model_epoch_{epoch+1}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")


import os
import shutil
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

class TripletTrainer:
    def __init__(self, model, train_dataloader, eval_dataloader, device, num_epochs, num_classes, lr=5e-5, alpha=1, beta=1, gamma=1):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_func = ContrastiveLoss()
        self.loss_func_classification = CrossEntropyLoss()
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.num_training_steps = num_epochs * len(train_dataloader)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps
        )
        self.best_val_loss = float("inf")
        self.train_loss_history = []
        self.val_loss_history = []

    def _train_epoch(self):
        self.model.train()
        total_loss = 0

        for sample1, sample2, sample3 in self.train_dataloader:
            sample1 = {k: v.to(self.device) for k, v in sample1.items()}
            sample2 = {k: v.to(self.device) for k, v in sample2.items()}
            sample3 = {k: v.to(self.device) for k, v in sample3.items()}
            
            out_x1, out_x2, out_x3, update_x1, update_x2, update_x3 = self.model(sample1, sample2, sample3)
            self.optimizer.zero_grad()

            loss_val = self.gamma * self.loss_func(out_x1, out_x2, out_x3)
            loss_val.backward(retain_graph=True)
            self.optimizer.step()

            x1_classification_val = self.alpha * self.loss_func_classification(update_x1)
            x1_classification_val.backward(retain_graph=True)
            self.optimizer.step()

            total_loss += loss_val.item() + x1_classification_val.item()
            self.lr_scheduler.step()

        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for sample1 in self.eval_dataloader:
                sample1 = {k: v.to(self.device) for k, v in sample1.items()}
                out_x1, update_x1 = self.model(sample1, None, None)
                x1_classification_val = self.loss_func_classification(update_x1)
                total_loss += x1_classification_val.detach().cpu().item()

        avg_loss = total_loss / len(self.eval_dataloader)
        return avg_loss

    def train(self):
        progress_bar = tqdm(range(self.num_training_steps), desc="Training")

        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            print(f"Epoch {epoch+1}/{self.num_epochs} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model_checkpoint(epoch)
                print("Saving checkpoint!")
            
            progress_bar.update(len(self.train_dataloader))

        print("Training complete.")
        return self.train_loss_history, self.val_loss_history

    def _save_model_checkpoint(self, epoch):
        checkpoint_path = f"best_triplet_model_epoch_{epoch+1}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
