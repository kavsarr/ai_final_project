import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy
from torchmetrics import F1Score
from torchvision import transforms

from torch.utils.data import DataLoader, Dataset
from models.resnet import ResNet18
from models.vgg import VGG16
from datasets.dataset_retrieval import custom_dataset
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
import tqdm

import os

save_model_path = "checkpoints/"


def val(model, data_val, loss_function, writer, epoch, device):
    f1score = 0
    f1 = F1Score(num_classes=62, task = 'multiclass')
    data_iterator = enumerate(data_val)  # take batches
    f1_list = []
    f1t_list = []

    with torch.no_grad():
        model.eval()  # switch model to evaluation mode
        tq = tqdm.tqdm(total=len(data_val))
        tq.set_description('Validation:')

        total_loss = 0

        for _, batch in data_iterator:
            # forward propagation
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            pred = model(image)

            loss = loss_function(pred, label.float())

            pred = pred.softmax(dim=1)
            
            f1_list.extend(torch.argmax(pred, dim =1).tolist())
            f1t_list.extend(torch.argmax(label, dim =1).tolist())

            total_loss += loss.item()
            tq.update(1)

    
    f1score = f1(torch.tensor(f1_list), torch.tensor(f1t_list))
    writer.add_scalar("Validation F1", f1score, epoch)
    writer.add_scalar("Validation Loss", total_loss/len(data_val), epoch)


    tq.close()
    print("F1 score: ", f1score)


    return None


def train(model, train_loader, val_loader, optimizer, loss_fn, n_epochs, device, log_dir):
  
    writer = SummaryWriter(log_dir=os.path.join('runs', log_dir))

    model.to(device)  # Move the model to the specified device (e.g., GPU or CPU)
    model.train()  # Set the model to training mode
    for epoch in range(n_epochs):
        
        
        model.train()
        running_loss = 0.0
        
        tq = tqdm.tqdm(total=len(train_loader))
        tq.set_description('epoch %d' % (epoch))
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)  # Move the batch of images to the specified device
            labels = labels.to(device)  # Move the batch of labels to the specified device
            
            optimizer.zero_grad()  # Reset the gradients of the optimizer
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = loss_fn(outputs, labels)
            outputs = outputs.softmax(dim=1)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()
            
            running_loss += loss.item()
            tq.set_postfix(loss_st='%.6f' % loss.item())
            tq.update(1)
        
        writer.add_scalar("Training Loss", running_loss/len(train_loader), epoch)
           
        tq.close()
        epoch_loss = running_loss / len(train_loader)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, epoch_loss))
                
        #check the performance of the model on unseen dataset4
        val(model, val_loader, loss_fn, writer, epoch, device)
        
        #save the model in pth format
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(save_model_path, log_dir+'.pth'))
        print("saved the model " + save_model_path)




def main(model_name, optimizer_name, lr, pretrained, log_dir):
    
    device = "cuda"
    
    tr_train = transforms.Compose([
        transforms.RandomRotation(45, fill=255),
        # transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3, fill=255),
        transforms.RandomAffine(degrees=0, scale=(0.6, 1.2), fill=255),
        transforms.Resize([250, 250]),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    tr_val = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    train_data = custom_dataset("train", transforms=tr_train)
    val_data = custom_dataset("val", transforms=tr_val)

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=32,
        drop_last=True
    )
    
    max_epoch = 25

    if model_name=='resnet':
        model = ResNet18(num_classes=62, pretrained=pretrained).to(device)
    elif model_name=='vgg':
        model = VGG16(num_classes=62, pretrained=pretrained).to(device)
    
    if optimizer_name=='sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum = 0.9)
    elif optimizer_name=='adam':
        optimizer = Adam(model.parameters(), lr=lr)
    
    loss = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, loss, max_epoch, device, log_dir)
    
    
if __name__ == "__main__":
    
    main(model_name='resnet', optimizer_name='sgd', lr=5e-3, pretrained=False, log_dir='resnet_sgd')
    main(model_name='resnet', optimizer_name='adam', lr=1e-4, pretrained=False, log_dir='resnet_adam')
    
    main(model_name='resnet', optimizer_name='sgd', lr=5e-4, pretrained=True, log_dir='resnet_sgd_pretrained')
    main(model_name='resnet', optimizer_name='adam', lr=5e-5, pretrained=True, log_dir='resnet_adam_pretrained')
    
    main(model_name='vgg', optimizer_name='sgd', lr=5e-3, pretrained=False, log_dir='vgg_sgd')
    main(model_name='vgg', optimizer_name='adam', lr=1e-4, pretrained=False, log_dir='vgg_adam')
    
    main(model_name='vgg', optimizer_name='sgd', lr=5e-4, pretrained=True, log_dir='vgg_sgd_pretrained')
    main(model_name='vgg', optimizer_name='adam', lr=5e-5, pretrained=True, log_dir='vgg_adam_pretrained')