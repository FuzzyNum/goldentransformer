import random
import logging
import numpy as np
import torch
import csv

class Ground_truth_model:
    def __init__(self,model,tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.original_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}  # Save original weights
    def preprocess(self, text):
        return self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    def predict(self, text):
        inputs = self.preprocess(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return torch.argmax(outputs.logits, dim=1).item()
    def train(self, train_dataloader, epochs=5, learning_rate=2e-5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_dataloader:
                input_ids, attention_mask, labels = (t.to(device) for t in batch)
                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
    def evaluate(self, dataloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        total, correct = 0, 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = (t.to(device) for t in batch)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy
    def reset_model_weights(self):
    
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(self.original_state_dict[name])
    