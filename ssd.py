import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc
import os


class SSD(nn.Module):
    """
    Defines the SSD-Multibox Architecture
    The network uses the base VGG network followed by the
    added multibox convolutional layers.  
    
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """
        Performs forward propagation on a given image
        """
        sources = list()
        loc = list()
        conf = list()

        for k in range(len(self.vgg)):
            x= self.vgg[k](x)
            if k==22:
                s = self.L2Norm(x)
                sources.append(s)

        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Use .pth or .pkl files only.')


def Maxpool_layer(ceil):
    if ceil==1:
        return [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
    else:
        return [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)]

def Conv_layer(in_channels,v, batch_norm):
    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
    if batch_norm:
        return [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
    else:
        return [conv2d, nn.ReLU(inplace=True)]

def Stack_i_Conv_Layer(in_channels,v,batch_norm,len_stack):
    stack_layer= nn.ModuleList()
    for i in range(len_stack):
        stack_layer += nn.ModuleList(Conv_layer(in_channels,v,batch_norm))
        in_channels=v

    return stack_layer
        

def vgg(i, batch_norm=False):
    """
    Models the base vgg-net.
    Derived from pytorch's vgg model.
    """
    layers = nn.ModuleList()
    in_channels = i
    layers += Stack_i_Conv_Layer(in_channels,64,batch_norm,2)
    layers += Maxpool_layer(0)
    in_channels = 64
    layers += Stack_i_Conv_Layer(in_channels,128,batch_norm,2)
    layers += Maxpool_layer(0) 
    in_channels = 128
    layers += Stack_i_Conv_Layer(in_channels,256,batch_norm,3)
    layers += Maxpool_layer(1) 
    in_channels = 256
    layers += Stack_i_Conv_Layer(in_channels,512,batch_norm,3)
    layers += Maxpool_layer(0) 
    in_channels = 512
    layers += Stack_i_Conv_Layer(in_channels,512,batch_norm,3)
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def extra_conv_layer(in_channels, v, s, pad):
    if s==1 and pad == 1:
        return [nn.Conv2d(in_channels, v, kernel_size=(3,3),stride=(2,2), padding=(1,1))]
    elif s==1 and pad==0:
        return [nn.Conv2d(in_channels, v, kernel_size=(3,3))]
    else:
        return [nn.Conv2d(in_channels, v, kernel_size=(1,1))]


def add_extras(i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = nn.ModuleList()
    in_channels = i
   
    layers += extra_conv_layer(in_channels, 256, 0, 0)
    in_channels=256
    layers += extra_conv_layer(in_channels, 512, len(layers)%2, 1)
    in_channels=512
    for i in range(3):
        layers += extra_conv_layer(in_channels, 128, 0, 0)
        in_channels=128
        if i==0:
            layers += extra_conv_layer(in_channels, 256, len(layers)%2, 1)
        else:
            layers += extra_conv_layer(in_channels, 256, len(layers)%2, 0)
        in_channels=256
    return layers


def multibox(vgg, extra_layers, num_classes):
    """
    Builds the final net
    """
    loc_layers = nn.ModuleList()
    conf_layers = nn.ModuleList()
    loc_layers += nn.ModuleList([nn.Conv2d(vgg[21].out_channels,
                                 4 * 4, kernel_size=3, padding=1)])
    conf_layers += nn.ModuleList([nn.Conv2d(vgg[21].out_channels,
                                 4 * num_classes, kernel_size=3, padding=1)])
    loc_layers += nn.ModuleList([nn.Conv2d(vgg[-2].out_channels,
                                 6 * 4, kernel_size=3, padding=1)])
    conf_layers += nn.ModuleList([nn.Conv2d(vgg[-2].out_channels,
                                  6 * num_classes, kernel_size=3, padding=1)])
    add_loc_stacked_layer_list = nn.ModuleList()
    add_loc_stacked_layer_list += nn.ModuleList([nn.Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]) 
    add_loc_stacked_layer_list += nn.ModuleList([nn.Conv2d(256, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))])
    add_loc_stacked_layer_list += nn.ModuleList([nn.Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))])
    add_loc_stacked_layer_list += nn.ModuleList([nn.Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))])

    add_conf_stacked_layer_list = nn.ModuleList()
    add_conf_stacked_layer_list += nn.ModuleList([nn.Conv2d(512, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))])
    add_conf_stacked_layer_list += nn.ModuleList([nn.Conv2d(256, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))])
    add_conf_stacked_layer_list += nn.ModuleList([nn.Conv2d(256, 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))])
    add_conf_stacked_layer_list += nn.ModuleList([nn.Conv2d(256, 84, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))])
    for i in range(1):
        loc_layers += add_loc_stacked_layer_list
        conf_layers += add_conf_stacked_layer_list

    return vgg, extra_layers, (loc_layers, conf_layers)


def build_ssd(phase, size=300, num_classes=21):
    """
    Builds the multibox net, by deriving layers from vgg.
    """
        
    base_, extras_, head_ = multibox(vgg(3),
                                     add_extras(1024),
                                     num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)

