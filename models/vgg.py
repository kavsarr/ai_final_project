import torch
import torch.nn as nn
import torchvision.models as model


class VGG16(nn.Module):
    
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
                
        self.vgg16 = model.vgg16(pretrained=pretrained)
        self.vgg16.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_classes, bias=True)

    def forward(self, image):
        out = self.vgg16(image)
        
        return out
    
    
if __name__=='__main__':
    model = VGG16(num_classes=47, pretrained=False)
    print(model)