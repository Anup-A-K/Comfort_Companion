import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class ConversationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = str(row['question'])
        answer = str(row['answer'])

        # Prepare inputs
        inputs = self.tokenizer.encode_plus(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Prepare targets
        targets = self.tokenizer.encode_plus(
            answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Get predictions
            pred = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(pred.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, predictions, actual_labels

def train_model():
    print("Loading dataset...")
    df = pd.read_csv('dataset/Cleaned_Conversation.csv')
    
    print("Initializing tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    # Create dataset
    full_dataset = ConversationDataset(df, tokenizer)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples")
    
    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 3
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in train_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            train_bar.set_postfix({'train_loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation phase
        print("\nRunning validation...")
        val_loss, predictions, actual_labels = evaluate_model(model, val_dataloader, device)
        
        # Print epoch metrics
        print(f"\nEpoch {epoch + 1} Metrics:")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("New best validation loss! Saving model...")
            output_dir = os.path.join(os.getcwd(), 'trained_model')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_model()
