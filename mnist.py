import numpy as np
import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import Subset
import os
# import skimage.io as io

class Backbone(nn.Module):
    def __init__(self):
        '''
        this is the backbone of MNIST with three layer, kernel limitaion:3x3x224
        '''
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        # self.fc = nn.Linear(in_features=61952, out_features=10) # make sure the input put feature map size
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_input_size = self._get_conv_output((1, 28, 28))
        self.fc = nn.Linear(self.fc_input_size, 10)
    def _get_conv_output(self, shape):
        # 創建一個虛擬張量以通過卷積層來計算輸出大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            x = self.conv1(dummy_input)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = F.relu(x)

            return int(torch.flatten(x, 1).size(1))
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = x3.view(-1, self.fc_input_size)  # Flatten the tensor
        output = F.log_softmax(self.fc(x4), dim=1)
       
        return output
    # def forward_for_eval(self, x)
        # x1 = self.conv1(x)
        # x1 = F.relu(x1)
        # x2 = self.conv2(x1)
        # x2 = F.relu(x2)
        # x3 = self.conv3(x2)
        # x3 = torch.flatten(x3, 1)
        # output = self.fc(x3)
        # output = self.fc(x4)

        # return output

def objective(trial):
    # Suggest values for hyperparameters
    args = parse_arguments()
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)

    # Load datasets with the suggested batch size
    train_dataloader, test_dataloader = load_datasets(batch_size=batch_size, fraction=0.1)

    # Initialize model, optimizer, and scheduler with the suggested learning rate
    model, optimizer = initialize_model_and_optimizer(lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    device = setup_device()

    # Train the model for a few epochs
    for epoch in range(5):  # Example using only 5 epochs for each trial
        train(args, train_dataloader, model, device, optimizer, scheduler, epoch)
    
    # Evaluate the model and get the validation accuracy
    accuracy = eval(args, test_dataloader, model, device)

    # Return the validation accuracy (or validation loss if optimizing for loss)
    return accuracy
    

def train(args, train_dataloader, model, device, optimizer, epoch):
    model.train()
    model = model.to(device)
    correct_predictions = 0
    total_samples = 0
    for _, data in enumerate(train_dataloader):
        image, gt = data                                # index depends on batch size index = prediction of y 
        image, gt = image.to(device), gt.to(device)
        optimizer.zero_grad()
        prediction = model(image)
        # print(prediction.shape, type(prediction))
        loss = torch.nn.functional.nll_loss(input=prediction, target=gt)
        loss.backward()
        optimizer.step()
    # Calculate number of correct predictions
        predicted_class = torch.argmax(prediction, dim=1)
        correct_predictions += (predicted_class == gt).sum().item()
        total_samples += gt.size(0)  # Add the number of samples in this batch
    
    
    accuracy = (correct_predictions / total_samples) * 100
    print(f"Epoch NO.{epoch}; Loss: {loss.item():.4f}; Training Accuracy: {accuracy:.2f}%")

def eval(args, test_dataloader, model, device):
    model.eval()
    model = model.to(device)
    loss_overall = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            image, gt = data
            image, gt = image.to(device), gt.to(device)
            prediction = model(image)
            loss = torch.nn.functional.nll_loss(input=prediction, target=gt)
            loss_overall += loss
            # accuracy
            predicted_class = torch.argmax(prediction, dim=1)  # Get the index of the max log-probability
            correct_predictions += (predicted_class == gt).sum().item()  # Count correct predictions
            total_samples += gt.size(0)  # Update the total number of samples

        loss_overall = loss_overall / len(test_dataloader)
        accuracy = (correct_predictions /total_samples) * 100
        print(f"Test Loss: {loss_overall:.4f}, Test Accuracy: {accuracy:.2f}%")




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="number of epochs", default=30, type=int)
    parser.add_argument("--batch", help="number of batch number", default=32, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.01, type=float)
    return parser.parse_args()

def load_datasets(batch_size, fraction=1):
    # Load MNIST dataset and init dataloader
    train_dataset = torchvision.datasets.MNIST(".", train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(".", train=False, download=True, transform=torchvision.transforms.ToTensor())
    num_samples = int(len(train_dataset) * fraction)
    
    # Create a random subset of the dataset
    indices = np.random.choice(len(train_dataset), num_samples, replace=False)
    train_subset = Subset(train_dataset, indices)
    train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_dataloader, test_dataloader

def setup_device():
    # Check if the GPU is available
    use_cuda = torch.cuda.is_available()
    print(f"using CUDA {use_cuda}")
    device = "cuda" if use_cuda else "cpu"
    
    return device

def initialize_model_and_optimizer(lr):
    model = Backbone()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    return model, optimizer

def main(training_mode=False, evaluation_model=True): 
    # Parse arguments
    args = parse_arguments()
    
    # Setup device
    device = setup_device()
    
    # Load datasets
    train_dataloader, test_dataloader = load_datasets(args.batch)
    
    # Initialize model and optimizer
    model, opt = initialize_model_and_optimizer(args.lr)
    opt = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    # Train and evaluate
    if training_mode:
        for epoch in range(args.epochs):
            train(args=args, train_dataloader=train_dataloader, model=model, device=device, optimizer=opt, epoch=epoch)
        
        # save the trained weight of model
        torch.save(model.state_dict(),"model_weights.pth")
        print("Model weights saved successfully.")
    if evaluation_model:
        if os.path.exists('model_weights.pth'):
            model.load_state_dict(torch.load('model_weights.pth'))
            model.to(device)
            print("Model weights loaded successfully for evaluation.")
            # evaluation
            eval(args=args, test_dataloader=test_dataloader, model=model, device=device)
        else:
            print("Model weights not found. Please train the model first")

if __name__ == "__main__":
    main(training_mode=False)











