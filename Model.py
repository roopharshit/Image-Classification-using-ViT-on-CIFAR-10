# Model.py
import torch
from Network import VisionTransformer
from DataLoader import get_dataloaders
from Configure import get_config
from tqdm import tqdm

class VisionTransformerModel(torch.nn.Module):
    def __init__(self, config):
        super(VisionTransformerModel, self).__init__()
        self.config = config
        self.model = VisionTransformer(
            image_size=config['image_size'], patch_size=config['patch_size'],
            in_channels=config['in_channels'], embed_dim=config['embed_dim'],
            num_heads=config['num_heads'], mlp_dim=config['mlp_dim'],
            num_layers=config['num_layers'], num_classes=config['num_classes'],
            dropout=config['dropout']
        ).to(config['device'])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        train_loader,finetune_loader, test_loader = get_dataloaders(config['batch_size'])
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=config['learning_rate'], steps_per_epoch=len(train_loader), epochs=config['num_epochs'])

    def train(self, train_loader, num_epochs):
        self.model.train()
        number_epochs = num_epochs
        for epoch in range(num_epochs):
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{number_epochs}')
            running_loss = 0.0
            running_accuracy = 0.0
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.config['device']), labels.to(self.config['device'])
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                acc = (outputs.argmax(dim=1) == labels).float().mean()
                running_loss += loss.item()
                running_accuracy += acc / len(train_loader)

                pbar.set_postfix({
                    'Training Loss': f'{running_loss / len(train_loader):.4f}',
                    'Training Accuracy': f'{running_accuracy * 100:.2f}%'
                })

    
    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.config['device']), labels.to(self.config['device'])
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
    def predict(self, data_loader):
        self.model.eval()
        all_probabilities = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(self.config['device'])
                outputs = self.model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to convert logits to probabilities
                all_probabilities.extend(probabilities.cpu().numpy())  # Move probabilities to CPU and convert to NumPy array
        return all_probabilities
    
    
    def save_model(self, path="C:/Users/rakes/OneDrive/Desktop/ViT/saved_models/saved_model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
