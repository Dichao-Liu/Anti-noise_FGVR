from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchvision.models
from sam import SAM
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import imgaug as ia
import imgaug.augmenters as iaa
from vic.loss import CharbonnierLoss
import numpy as np
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from basic_conv import *
from example.model.smooth_cross_entropy import smooth_crossentropy
from example.utility.bypass_bn import enable_running_stats, disable_running_stats

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

def test(net, criterion, batch_size, test_path):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda")

    transform_test = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(root=test_path,
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        output_1, output_2, output_3, output_ORI, map1, map2, map3 = net(inputs)

        outputs_com = output_1 + output_2 + output_3 + output_ORI

        loss = criterion(output_ORI, targets).mean()

        test_loss += loss.item()
        _, predicted = torch.max(output_ORI.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct_com += predicted_com.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0:
            print('Step: %d | Loss: %.3f |Combined Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1),
            100. * float(correct_com) / total, correct_com, total))

    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc_en, test_loss




class Features(nn.Module):
    def __init__(self, net_layers):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers[0])
        self.net_layer_1 = nn.Sequential(net_layers[1])
        self.net_layer_2 = nn.Sequential(net_layers[2])
        self.net_layer_3 = nn.Sequential(net_layers[3])
        self.net_layer_4 = nn.Sequential(*net_layers[4])
        self.net_layer_5 = nn.Sequential(*net_layers[5])
        self.net_layer_6 = nn.Sequential(*net_layers[6])
        self.net_layer_7 = nn.Sequential(*net_layers[7])


    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x = self.net_layer_3(x)
        x = self.net_layer_4(x)
        x1 = self.net_layer_5(x)
        x2 = self.net_layer_6(x1)
        x3 = self.net_layer_7(x2)
        return x1, x2, x3


class Anti_Noise_Decoder(nn.Module):
    def __init__(self, scale, in_channel):
        super(Anti_Noise_Decoder, self).__init__()
        self.Sigmoid = nn.Sigmoid()

        in_channel = in_channel // (scale * scale)

        self.skip = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)

        )

        self.process = nn.Sequential(
            nn.PixelShuffle(scale),
            nn.Conv2d(in_channel, 256, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 3, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x, map):
        return self.skip(x) + self.process(map)

class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_class, classifier):
        super().__init__()
        self.Features = Features(net_layers)
        self.classifier_pool = nn.Sequential(classifier[0])
        self.classifier_initial = nn.Sequential(classifier[1])
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.max_pool1 = nn.MaxPool2d(kernel_size=56, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=28, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=14, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )


    def forward(self, x):
        x1, x2, x3 = self.Features(x)
        map1 = x1.clone()
        map2 = x2.clone()
        map3 = x3.clone()

        classifiers = self.classifier_pool(x3).view(x3.size(0), -1)
        classifiers = self.classifier_initial(classifiers)

        x1_ = self.conv_block1(x1)
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)
        x1_c = self.classifier1(x1_f)

        x2_ = self.conv_block2(x2)
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)


        x3_ = self.conv_block3(x3)
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        return x1_c, x2_c, x3_c, classifiers, map1, map2, map3

def img_add_noise(x, transformation_seq):

    x = x.permute(0, 2, 3, 1)
    x = x.cpu().numpy()
    x = transformation_seq(images=x)
    x = torch.from_numpy(x.astype(np.float32))
    x = x.permute(0, 3, 1, 2)
    return x


def CELoss(x, y):
    return smooth_crossentropy(x, y, smoothing=0.1)


def inference(batch_size=3, model_path='', num_class=0,  data_path='', use_state_dict = False):

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    if use_state_dict:
        net = torchvision.models.resnet50()
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
        net.load_state_dict(state_dict)
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features, num_class)
        net_layers = list(net.children())
        classifier = net_layers[8:10]
        net_layers = net_layers[0:8]
        net = Network_Wrapper(net_layers, num_class, classifier)
        net.load_state_dict(torch.load(model_path))
    else:
        net = torch.load(model_path)

    device = torch.device("cuda")
    net.to(device)
    
    val_acc_com, val_loss = test(net, CELoss, batch_size, data_path+'/test')
    print("Validation Accuracy (%): {} | Validation Loss: {}".format(val_acc_com, val_loss))



if __name__ == '__main__':
    data_path = '<The-path-To>/Stanford Cars'
    
    # set model_path as:
    # model_path='<The-path-To>/Stanford_Cars_ResNet50_Teacher_Network.pth', or
    # model_path='<The-path-To>/Stanford_Cars_ResNet50_Teacher_Weight.pth'
    model_path = ""
    
    model_path_file = model_path.split('/')
    model_path_file = model_path_file[-1]
    if 'Weight' in model_path_file:
        use_state_dict = True
    elif 'Network' in model_path_file:
        use_state_dict = False
    else:
        raise Exception("Unknown Model "+model_path_file)
    
    inference(batch_size=7,         
              model_path=model_path,     
              num_class=196,         
              data_path = data_path,
              use_state_dict = use_state_dict)  

