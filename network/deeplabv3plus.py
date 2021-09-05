import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

class _SimpleSegmentationContrastiveModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationContrastiveModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        features = self.backbone(x)
        x = self.classifier(features)
        return x

class _SimpleSegmentationPIRLModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationPIRLModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x, x_jig):
        features_x = self.backbone(x)
        features_x_jig = self.backbone(x_jig)
        x = self.classifier(features_x, features_x_jig)
        return x

class IntermediateLayerGetter(nn.ModuleDict):

    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

class Contrastive_head(nn.Module):
    def __init__(self, in_channels, classes):
        super(Contrastive_head, self).__init__()
        self.project = nn.Sequential( 
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, classes),
        )

        self._init_weight()

    def forward(self, feature):
        output_feature = self.project(feature['out'])
        return output_feature
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class PIRL_head(nn.Module):
    def __init__(self, in_channels, classes):
        super(PIRL_head, self).__init__()
        self.project = nn.Sequential( 
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, classes),
        )

        self.project_jig = JigsawHead(dim_in=in_channels, dim_out=classes, head='mlp')

        self._init_weight()

    def forward(self, feature_x, feature_x_jig):
        out = self.project(feature_x['out'])
        out_jig = self.project_jig(feature_x_jig['out'])
        return out, out_jig
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module

# Self-Supervised Learning of Pretext-Invariant Representations: https://github.com/HobbitLong/PyContrast/tree/master/pycontrast
class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)

class JigsawHead(nn.Module):
    """Jigswa + linear + l2norm"""
    def __init__(self, dim_in, dim_out, k=9, head='linear'):
        super(JigsawHead, self).__init__()

        if head == 'linear':
            self.fc1 = nn.Sequential([
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(dim_in, dim_out)
            ])
        elif head == 'mlp':
            self.fc1 = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, dim_out),
            )
        else:
            raise NotImplementedError('JigSaw head not supported: {}'.format(head))
        self.fc2 = nn.Linear(dim_out * k, dim_out)
        self.l2norm = Normalize(2)
        self.k = k

    def forward(self, x):
        bsz = x.shape[0]
        x = self.fc1(x)
        # ==== shuffle ====
        # this step can be moved to data processing step
        shuffle_ids = self.get_shuffle_ids(bsz)
        x = x[shuffle_ids]
        # ==== shuffle ====
        n_img = int(bsz / self.k)
        x = x.view(n_img, -1)
        x = self.fc2(x)
        x = self.l2norm(x)
        return x

    def get_shuffle_ids(self, bsz):
        n_img = int(bsz / self.k)
        rnd_ids = [torch.randperm(self.k) for i in range(n_img)]
        rnd_ids = torch.cat(rnd_ids, dim=0)
        base_ids = torch.arange(bsz)
        base_ids = torch.div(base_ids, self.k).long()
        base_ids = base_ids * self.k
        shuffle_ids = rnd_ids + base_ids
        return shuffle_ids

from . import resnet_deeplab as resnet
# import resnet_deeplab as resnet

def deeplabv3_resnet50(num_classes=3, pretrained_backbone=False):
    replace_stride_with_dilation=[False, False, True]
    aspp_dilate = [6, 12, 18]
    backbone = resnet.__dict__['resnet50'](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    inplanes = 2048
    low_level_planes = 256
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = _SimpleSegmentationModel(backbone, classifier)
    return model

def deeplabv3_resnet50_contrast(classes=512, pretrained_backbone=False):
    replace_stride_with_dilation=[False, False, True]
    backbone = resnet.__dict__['resnet50'](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    inplanes = 2048
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = Contrastive_head(inplanes, classes)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = _SimpleSegmentationContrastiveModel(backbone, classifier)
    return model

def deeplabv3_resnet50_classification(classes=4, pretrained_backbone=False):
    replace_stride_with_dilation=[False, False, True]
    backbone = resnet.__dict__['resnet50'](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    inplanes = 2048
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = Contrastive_head(inplanes, classes)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = _SimpleSegmentationContrastiveModel(backbone, classifier)
    return model

def deeplabv3_resnet50_pirl(classes=512, pretrained_backbone=False):
    replace_stride_with_dilation=[False, False, True]
    backbone = resnet.__dict__['resnet50'](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    inplanes = 2048
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = PIRL_head(inplanes, classes)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = _SimpleSegmentationPIRLModel(backbone, classifier)
    return model
