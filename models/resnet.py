import torch
import torch.nn as nn
import torchvision.models as model


class ResNet18(nn.Module):
    
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
                
        self.resnet18 = model.resnet18(pretrained=pretrained)
        self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-1]))    
        
        self.classifier = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)


    def forward(self, image):
        resnet_pred = self.resnet18(image).squeeze()
        out = self.classifier(resnet_pred)
        
        return out
    
    
if __name__=='__main__':
    model = model.resnet18(pretrained=False)
    print(model)