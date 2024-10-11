from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torchvision.models
from sam import SAM
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import cv2

import numpy as np
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from basic_conv import *
from example.model.smooth_cross_entropy import smooth_crossentropy
from example.utility.bypass_bn import enable_running_stats, disable_running_stats

import argparse
parser = argparse.ArgumentParser(description='train_student')
parser.add_argument('--from_local', required=False, action='store_true', 
                    help="By default, this is set to use the pre-trained parameters downloaded from the cloud as the teacher model. "
                         "Set this flag to True to use the locally trained parameters instead.")

args, unparsed = parser.parse_known_args()


class Student_Wrapper(nn.Module):
    def __init__(self, net_layers, classifier):
        super(Student_Wrapper, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers[0])
        self.net_layer_1 = nn.Sequential(net_layers[1])
        self.net_layer_2 = nn.Sequential(net_layers[2])
        self.net_layer_3 = nn.Sequential(net_layers[3])
        self.net_layer_4 = nn.Sequential(*net_layers[4])
        self.net_layer_5 = nn.Sequential(*net_layers[5])
        self.net_layer_6 = nn.Sequential(*net_layers[6])
        self.net_layer_7 = nn.Sequential(*net_layers[7])

        self.net_layer_8 = nn.Sequential(classifier[0])
        self.net_layer_9 = nn.Sequential(classifier[1])

        self.bn1 = nn.Sequential(nn.BatchNorm2d(64))
        self.relu = nn.Sequential(nn.ReLU())

    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x = self.net_layer_3(x)
        x = self.net_layer_4(x)
        x1 = self.net_layer_5(x)
        x2 = self.net_layer_6(x1)
        x3 = self.net_layer_7(x2)

        feat = self.net_layer_8(x3)
        feat = feat.view(feat.size(0), -1)
        out = self.net_layer_9(feat)

        return out, x1, x2, x3


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def test(net, criterion, batch_size, test_path):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
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
        output, _, _, _ = net(inputs)

        loss = criterion(output, targets).mean()

        test_loss += loss.item()
        _, predicted = torch.max(output.data, 1)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0:
            print('Step: %d | Loss: %.3f |Combined Acc: %.3f%% (%d/%d)' % (
                batch_idx, test_loss / (batch_idx + 1),
                100. * float(correct) / total, correct, total))

    test_acc_en = 100. * float(correct) / total
    test_loss = test_loss / (idx + 1)

    net.train()

    return test_acc_en, test_loss


def show_im(image, h, w):
    img = (np.transpose(image.cpu().detach().numpy(), (1, 2, 0)) * (0.5, 0.5, 0.5) + (0.5, 0.5, 0.5))

    img = cv2.resize(img, (w, h))
    return img


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


def train(nb_epoch, batch_size, store_name, num_class=1, start_epoch=0, data_path=''):
    MSE = nn.MSELoss()

    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root=data_path + '/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    net = torchvision.models.resnet50()
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    net.load_state_dict(state_dict)
    fc_features = net.fc.in_features
    net.fc = nn.Linear(fc_features, num_class)

    net_layers = list(net.children())
    classifier = net_layers[8:10]
    net_layers = net_layers[0:8]
    
    if not args.from_local:
        net_teacher = torch.load('weightsFromCloud/Stanford_Cars_ResNet50_Teacher_Network.pth')
    else:
        net_teacher = torch.load('results/Stanford_Cars_ResNet50_PMAL/model.pth')
        

    net_student = Student_Wrapper(net_layers, classifier)

    base_optimizer = torch.optim.SGD

    optimizer = SAM(net_student.parameters(), base_optimizer, lr=0.002, momentum=0.9, weight_decay=5e-4)

    device = torch.device("cuda")
    net_student.to(device)
    net_teacher.to(device)

    max_val_acc = 0
    lr = [0.002]
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)

        if epoch < (nb_epoch * (1/2)):
            net_student.train()
            train_loss = 0
            train_loss1 = 0
            train_loss2 = 0
            train_loss3 = 0
            train_loss4 = 0
            correct = 0
            total = 0
            idx = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                idx = batch_idx
                if inputs.shape[0] < batch_size:
                    continue
                if use_cuda:
                    inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = Variable(inputs), Variable(targets)

                for nlr in range(len(optimizer.param_groups)):
                    optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

                net_teacher.eval()
                with torch.no_grad():
                    output_1_t, output_2_t, output_3_t, output_4_t, map1_t, map2_t, map3_t = net_teacher(inputs)



                # h1
                # h1 first forward-backward step
                enable_running_stats(net_student)
                optimizer.zero_grad()
                _, x1, _, _ = net_student(inputs)
                loss1 = MSE(map1_t.detach(), x1) * 100
                loss1.backward()
                optimizer.first_step(zero_grad=True)
                # h1 second forward-backward step
                disable_running_stats(net_student)
                optimizer.zero_grad()
                _, x1, _, _ = net_student(inputs)
                loss1_ = MSE(map1_t.detach(), x1) * 100
                loss1_.backward()
                optimizer.second_step(zero_grad=True)

                # h2
                # h2 first forward-backward step
                enable_running_stats(net_student)
                optimizer.zero_grad()
                _, _, x2, _ = net_student(inputs)
                loss2 = MSE(map2_t.detach(), x2) * 100
                loss2.backward()
                optimizer.first_step(zero_grad=True)

                # h2 second forward-backward step
                disable_running_stats(net_student)
                optimizer.zero_grad()
                _, _, x2, _ = net_student(inputs)
                loss2_ = MSE(map2_t.detach(), x2) * 100
                loss2_.backward()
                optimizer.second_step(zero_grad=True)


                # h3
                # h3 first forward-backward step
                enable_running_stats(net_student)
                optimizer.zero_grad()
                _, _, _, x3 = net_student(inputs)
                loss3 = MSE(map3_t.detach(), x3) * 100
                loss3.backward()
                optimizer.first_step(zero_grad=True)
                # h3 second forward-backward step
                disable_running_stats(net_student)
                optimizer.zero_grad()
                _, _, _, x3 = net_student(inputs)
                loss3_ = MSE(map3_t.detach(), x3) * 100
                loss3_.backward()
                optimizer.second_step(zero_grad=True)



                # h4
                # h4 first forward-backward step
                enable_running_stats(net_student)
                optimizer.zero_grad()
                output, _, _, _ = net_student(inputs)

                loss4 = MSE(output_1_t.detach(), output) + \
                        MSE(output_2_t.detach(), output) + \
                        MSE(output_3_t.detach(), output) + \
                        MSE(output_4_t.detach(), output) + \
                        CELoss(output, targets).mean()
                loss4.backward()
                optimizer.first_step(zero_grad=True)
                # h4 second forward-backward step
                disable_running_stats(net_student)
                optimizer.zero_grad()
                output, _, _, _ = net_student(inputs)

                loss4_ = MSE(output_1_t.detach(), output) + \
                        MSE(output_2_t.detach(), output) + \
                        MSE(output_3_t.detach(), output) + \
                        MSE(output_4_t.detach(), output) + \
                        CELoss(output, targets).mean()
                loss4_.backward()
                optimizer.second_step(zero_grad=True)

                _, predicted = torch.max(output.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                train_loss += (loss1.item() + loss2.item() + loss3.item() + loss4.item())
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
                train_loss3 += loss3.item()
                train_loss4 += loss4.item()

                if batch_idx % 50 == 0:
                    print(
                        'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f |Loss4: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                            batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                            train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                            100. * float(correct) / total, correct, total))
        else:
            net_student.train()
            train_loss = 0
            correct = 0
            total = 0
            idx = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                idx = batch_idx
                if inputs.shape[0] < batch_size:
                    continue
                if use_cuda:
                    inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = Variable(inputs), Variable(targets)

                for nlr in range(len(optimizer.param_groups)):
                    optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

                #Fine-tune: first forward-backward step
                enable_running_stats(net_student)
                optimizer.zero_grad()
                output, _, _, _ = net_student(inputs)
                loss_f = CELoss(output, targets).mean()
                loss_f.backward()
                optimizer.first_step(zero_grad=True)

                # Fine-tune: second forward-backward step
                disable_running_stats(net_student)
                optimizer.zero_grad()
                output, _, _, _ = net_student(inputs)
                loss_f = CELoss(output, targets).mean()
                loss_f.backward()
                optimizer.second_step(zero_grad=True)



                _, predicted = torch.max(output.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                train_loss += loss_f.item()

                if batch_idx % 50 == 0:
                    print(
                        'Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                            batch_idx, train_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

        if epoch < (nb_epoch * (1/2)):
            train_acc = 100. * float(correct) / total
            train_loss = train_loss / (idx + 1)
            with open(exp_dir + '/results_train.txt', 'a') as file:
                file.write(
                    'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss4: %.5f |\n' % (
                        epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1),
                        train_loss3 / (idx + 1),
                        train_loss4 / (idx + 1)))
        else:
            train_acc = 100. * float(correct) / total
            train_loss = train_loss / (idx + 1)
            with open(exp_dir + '/results_train.txt', 'a') as file:
                file.write(
                    'Iteration %d | train_acc = %.5f | train_loss = %.5f|\n' % (epoch, train_acc, train_loss))


        val_acc_com, val_loss = test(net_student, CELoss, 7, data_path + '/test')
        if val_acc_com > max_val_acc:
            max_val_acc = val_acc_com
            net_student.cpu()
            torch.save(net_student, './' + store_name + '/model.pth')

            net_student.to(device)

        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write('Iteration %d, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc_com, val_loss))



if __name__ == '__main__':
    data_path = '<The-Path-To>/Stanford Cars'
    train(nb_epoch=200,  # number of epoch
          batch_size=8,  # batch size
          store_name='results/Stanford_Cars_ResNet50_Distillation',  # the folder for saving results
          num_class=196,  # number of categories
          start_epoch=0,  # the start epoch number
          data_path=data_path)  # the path to the dataset
