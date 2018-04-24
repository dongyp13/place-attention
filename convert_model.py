import torch
from torch.autograd import Variable as V
import torchvision.models as models
import attentionresnet
from PIL import Image
from torchvision import transforms as trn
from torch.nn import functional as F
import os

# th architecture to use
arch = 'resnet18'
model_weight = 'attentionresnet_best.pth.tar'
model_name = 'attentionresnet'

# create the network architecture
#model = models.__dict__[arch](num_classes=365)
model = attentionresnet.resnet18(num_classes=365)
#model_weight = '%s_places365.pth.tar' % arch

checkpoint = torch.load(model_weight, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()} # the data parallel layer will add 'module' before each layer name
model.load_state_dict(state_dict)
model.eval()

model.cpu()
torch.save(model, 'whole_' + model_name + '.pth.tar')
print('save to ' + 'whole_' + model_name + '.pth.tar')
