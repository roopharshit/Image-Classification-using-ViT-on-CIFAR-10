# main.py
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from Model import VisionTransformerModel
from DataLoader import get_dataloaders, apply_test_transforms
from Configure import get_config

def main():
    config = get_config()
    model = VisionTransformerModel(config)
    train_loader, finetune_loader, test_loader = get_dataloaders(config['batch_size'])
    
    test_images = np.load('C:/Users/rakes/OneDrive/Desktop/ViT/data2024/private_test_images_2024.npy')
    test_images = test_images.reshape(-1, 3, 32, 32)
    test_images = torch.tensor(test_images).float()  # Convert NumPy array to torch.Tensor and set type
    #test_images = test_images.view(-1, 3, 32, 32)  # Reshape assuming the data is in [N, 3072] shape
    
    # transformed_test_images = torch.stack([apply_test_transforms(img) for img in test_images])
    # test_dataset = torch.utils.data.TensorDataset(transformed_test_images)
    # public_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    test_images = apply_test_transforms(test_images)
    
    test_dataset = TensorDataset(test_images)
    public_test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print("Initial Training on the larger subset")
    model.train(train_loader, 250)
    
    print("Fine-tuning on smaller subset")
    model.train(finetune_loader, 10)
    model.save_model("C:/Users/rakes/OneDrive/Desktop/ViT/saved_models/saved_model.pth")
    
    print("Evaluating on test set")
    model.evaluate(test_loader)
    
    
    # Make predictions
    predictions = model.predict(public_test_loader)
    np.save('C:/Users/rakes/OneDrive/Desktop/ViT/saved_models/public_test_predictions.npy', predictions)
    print("Predictions saved succesfully")

if __name__ == "__main__":
    main()
