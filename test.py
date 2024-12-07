import torch
from torchmetrics import F1Score, Accuracy
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.dataset_retrieval import custom_dataset
from torch.utils.tensorboard import SummaryWriter
import tqdm
import os
from models.resnet import ResNet18
from models.vgg import VGG16


def test(model, test_loader, device):
    
    f1 = F1Score(num_classes=62, task='multiclass')
    acc = Accuracy(num_classes=62, task='multiclass')

    y_test = []
    y_pred = []

    with torch.no_grad():
        
        model.eval()
        
        tq = tqdm.tqdm(total=len(test_loader))
        tq.set_description('Testing:')
        
        data_iterator = enumerate(test_loader)

        for _, batch in data_iterator:

            image, label = batch
            image = image.to(device)
            label = label.to(device)
            
            pred = model(image)
            pred = pred.softmax(dim=1)

            y_test.extend(torch.argmax(label, dim=1).tolist())
            y_pred.extend(torch.argmax(pred, dim=1).tolist())
            
            tq.update(1)

        f1_score = f1(torch.tensor(y_pred), torch.tensor(y_test))
        acc_score = acc(torch.tensor(y_pred), torch.tensor(y_test))

        tq.close()
        print(f"F1: {f1_score} Accuracy: {acc_score}")

    return None


def main(model_name, log_dir):
    
    device = "cuda"
    
    tr_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_data = custom_dataset("test", transforms=tr_test)

    test_loader = DataLoader(
        test_data,
        batch_size=32,
        drop_last=True
    )

    if model_name == 'resnet':
        model = ResNet18(num_classes=62, pretrained=False).to(device)
    elif model_name == 'vgg':
        model = VGG16(num_classes=62, pretrained=False).to(device)
    
    checkpoint_path = os.path.join('checkpoints', log_dir+'.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    print(f"Loaded model from {checkpoint_path}")

    test(model, test_loader, device)


if __name__ == "__main__":
    main(model_name='resnet', log_dir='resnet_sgd')
    main(model_name='resnet', log_dir='resnet_adam')
    
    main(model_name='resnet', log_dir='resnet_sgd_pretrained')
    main(model_name='resnet', log_dir='resnet_adam_pretrained')
    
    main(model_name='vgg', log_dir='vgg_sgd')
    main(model_name='vgg', log_dir='vgg_adam')
    
    main(model_name='vgg', log_dir='vgg_sgd_pretrained')
    main(model_name='vgg', log_dir='vgg_adam_pretrained')