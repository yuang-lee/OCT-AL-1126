import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        
        if base_model == 'resnet18':
            # To train from random init, we need to modify here!!
            model = models.resnet18(weights='IMAGENET1K_V1')
        elif base_model == 'resnet18_random':
            model = models.resnet18()
        elif base_model == 'resnet34':
            model = models.resnet34(weights='IMAGENET1K_V1')
        elif base_model == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V2')
        elif base_model == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V2')
        elif base_model == 'resnet152':
            model = models.resnet152(weights='IMAGENET1K_V2')
        num_feat = model.fc.in_features
        model.fc = nn.Linear(num_feat,out_dim)
        self.resnet_dict = {base_model: model}
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)

