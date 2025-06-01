import torch
import argparse
from pathlib import Path

from models import resnet20, resnet32, resnet56, resnet20_slim, resnet32_slim
from dataset import get_dataloaders

def test_model(model_path, model_type, data_dir='./data', batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_dict = {
        'resnet20': resnet20,
        'resnet32': resnet32,
        'resnet56': resnet56,
        'resnet20_slim': resnet20_slim,
        'resnet32_slim': resnet32_slim
    }
    
    model = model_dict[model_type]().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size, augment=False)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Best training accuracy: {checkpoint["best_acc"]:.2f}%')
    
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['resnet20', 'resnet32', 'resnet56', 'resnet20_slim', 'resnet32_slim'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    
    test_model(args.model_path, args.model_type, args.data_dir, args.batch_size)

if __name__ == '__main__':
    main()