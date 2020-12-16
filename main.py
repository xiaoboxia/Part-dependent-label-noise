# -*- coding: utf-8 -*-

import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import LeNet, Matrix_optimize
import torchvision.transforms as transforms
import numpy as np
import argparse
import datetime
import resnet
import tools
import data
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--lr_revision', type = float, default = 5e-7)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--model_dir', type=str, help='dir to save model files', default='model/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--noise_type', type = str, help='[instance, symmetric]', default='instance')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, cifar100', default = 'mnist')
parser.add_argument('--n_epoch_1', type = int, help = 'estimate', default=10)
parser.add_argument('--n_epoch_2', type = int, help = 'loss correction',default=100)
parser.add_argument('--n_epoch_3', type = int, help = 'revision',default=50)
parser.add_argument('--n_epoch_4', type = int, help = 'learn matrix',default=1500)
parser.add_argument('--iteration_nmf', type = int, default=20)
parser.add_argument('--optimizer', type = str, default='SGD')
parser.add_argument('--seed', type = int, default=5)
parser.add_argument('--print_freq', type = int, default=100)
parser.add_argument('--num_workers', type = int, default=8, help='how many subprocesses to use for data loading')
parser.add_argument('--model_type', type = str, help='[ce, ours]', default='ours')
parser.add_argument('--split_percentage', type = float, help = 'train and validation', default=0.9)
parser.add_argument('--norm_std', type = float, help = 'distribution ', default=0.1)
parser.add_argument('--num_classes', type = int, help = 'num_classes', default=10)
parser.add_argument('--feature_size', type = int, help = 'the size of feature_size', default=784)
parser.add_argument('--dim', type = int, help = 'the dim of representations', default=84)
parser.add_argument('--basis', type = int, help = 'the num of basis', default=10)
parser.add_argument('--weight_decay', type = float, help = 'weight', default=1e-4)
parser.add_argument('--momentum', type = float, help = 'momentum', default=0.9)
parser.add_argument('--gpu', type = int, help = 'ind of gpu', default=0)
args = parser.parse_args()
#
torch.cuda.set_device(args.gpu)
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr 

# load dataset
if args.dataset=='mnist':
    args.feature_size = 28 * 28
    args.num_classes = 10
    args.n_epoch_1, args.n_epoch_2, args.n_epoch_3 = 5, 20, 50
    args.dim = 84
    args.basis = 10
    train_dataset = data.mnist_dataset(True,
                                    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307, ),(0.3081, )),]),
                                    target_transform=tools.transform_target,
                                    noise_rate=args.noise_rate,
                                    split_percentage=args.split_percentage,
                                    seed=args.seed)
    
    val_dataset = data.mnist_dataset(False,
                                    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307, ),(0.3081, )),]),
                                    target_transform=tools.transform_target,
                                    noise_rate=args.noise_rate,
                                    split_percentage=args.split_percentage,
                                    seed=args.seed)


    test_dataset =  data.mnist_test_dataset(
                                    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307, ),(0.3081, )),]),
                                    target_transform=tools.transform_target)
    
    
    
if args.dataset=='fashionmnist':
    args.feature_size = 28 * 28
    args.num_classes = 10
    args.n_epoch_1, args.n_epoch_2, args.n_epoch_3 = 5, 20, 50
    args.dim = 512
    args.basis = 10
    train_dataset = data.fashionmnist_dataset(True,
                                    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307, ),(0.3081, )),]),
                                    target_transform=tools.transform_target,
                                    noise_rate=args.noise_rate,
                                    split_percentage=args.split_percentage,
                                    seed=args.seed)
    
    val_dataset = data.fashionmnist_dataset(False,
                                    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307, ),(0.3081, )),]),
                                    target_transform=tools.transform_target,
                                    noise_rate=args.noise_rate,
                                    split_percentage=args.split_percentage,
                                    seed=args.seed)


    test_dataset =  data.fashionmnist_test_dataset(
                                    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307, ),(0.3081, )),]),
                                    target_transform=tools.transform_target)  
if args.dataset=='cifar10':
    args.num_classes = 10
    args.feature_size = 3 * 32 * 32
    args.n_epoch_1, args.n_epoch_2, args.n_epoch_3 = 5, 50, 50
    args.dim = 512
    args.basis = 20
    args.iteration_nmf = 10
    train_dataset = data.cifar10_dataset(True,
                                    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                    ]),
                                    target_transform=tools.transform_target,
                                    noise_rate=args.noise_rate,
                                    split_percentage=args.split_percentage,
                                    seed=args.seed)
    
    val_dataset = data.cifar10_dataset(False,
                                    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                    ]),
                                    target_transform=tools.transform_target,
                                    noise_rate=args.noise_rate,
                                    split_percentage=args.split_percentage,
                                    seed=args.seed)


    test_dataset =  data.cifar10_test_dataset(
                                    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                    ]),
                                    target_transform=tools.transform_target)

if args.dataset=='svhn':
    args.num_classes = 10
    args.feature_size = 3 * 32 * 32
    args.n_epoch_1, args.n_epoch_2, args.n_epoch_3 = 5, 50, 50
    args.dim = 512
    args.basis = 10
    train_dataset = data.svhn_dataset(True,
                                    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
                                    ]),
                                    target_transform=tools.transform_target,
                                    noise_rate=args.noise_rate,
                                    split_percentage=args.split_percentage,
                                    seed=args.seed)
    
    val_dataset = data.svhn_dataset(False,
                                    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
                                    ]),
                                    target_transform=tools.transform_target,
                                    noise_rate=args.noise_rate,
                                    split_percentage=args.split_percentage,
                                    seed=args.seed)


    test_dataset =  data.svhn_test_dataset(
                                    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
                                    ]),
                                    target_transform=tools.transform_target)
    

# mkdir 
model_save_dir = args.model_dir + '/' + args.dataset + '/' + 'noise_rate_%s'%(args.noise_rate)

if not os.path.exists(model_save_dir):
    os.system('mkdir -p %s'%(model_save_dir))
  
save_dir = args.result_dir +'/' +args.dataset+'/%s/' % args.model_type

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str = args.dataset + '_%s_' % args.model_type + args.noise_type + '_' + str(args.noise_rate) 

txtfile = save_dir + "/" + model_str + ".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


def norm(T):
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm


def train_m(V, r, k, e):

    m, n = np.shape(V)
    W = np.mat(np.random.random((m, r)))
    H = np.mat(np.random.random((r, n)))
    data = []
    
    for x in range(k):
        V_pre = np.dot(W, H)
        E = V - V_pre
        err = 0.0
        err = np.sum(np.square(E))
        data.append(err)
        if err < e:  # threshold
            break

        a = np.dot(W.T, V)  # Hkj
        b = np.dot(np.dot(W.T, W), H)

        for i_1 in range(r):
            for j_1 in range(n):
                if b[i_1, j_1] != 0:
                    H[i_1, j_1] = H[i_1, j_1] * a[i_1, j_1] / b[i_1, j_1]

        c = np.dot(V, H.T)
        d = np.dot(np.dot(W, H), H.T)
        for i_2 in range(m):
            for j_2 in range(r):
                if d[i_2, j_2] != 0:
                    W[i_2, j_2] = W[i_2, j_2] * c[i_2, j_2] / d[i_2, j_2]



        W = norm(W)


    return W, H, data

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model

def train(model, train_loader, epoch, optimizer, criterion):
    print('Training %s...' % model_str)
    
    train_total=0
    train_correct=0 
   

    for i, (data, labels) in enumerate(train_loader):
        data = data.cuda()
        labels = labels.cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        _, logits=model(data, revision=False)
        prec1,  = accuracy(logits, labels, topk=(1, ))
        train_total+=1
        train_correct+=prec1
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f' 
                  %(epoch+1, args.n_epoch_1, i+1, len(train_dataset)//batch_size, prec1, loss.item()))
        
    train_acc=float(train_correct)/float(train_total)
   
    return train_acc

def train_correction(model, train_loader, epoch, optimizer, W_group, basis_matrix_group, batch_size, num_classes, basis):
    print('Training %s...' % model_str)
    
    train_total=0
    train_correct=0

    for i, (data, labels) in enumerate(train_loader):
        loss = 0.
        data = data.cuda()
        labels = labels.cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        _, logits=model(data, revision=False)
        
        logits_ = F.softmax(logits, dim=1)
        logits_correction_total = torch.zeros(len(labels), num_classes)
        for j in range(len(labels)):
            idx = i * batch_size + j
            matrix = matrix_combination(basis_matrix_group, W_group, idx, num_classes, basis)
            matrix = torch.from_numpy(matrix).float().cuda()
            logits_single = logits_[j, :].unsqueeze(0)
            logits_correction = logits_single.mm(matrix)
            pro1 = logits_single[:, labels[j]]
            pro2 = logits_correction[:, labels[j]]
            beta = Variable(pro1/pro2, requires_grad=True)
            logits_correction = torch.log(logits_correction+1e-12)
            logits_single = torch.log(logits_single + 1e-12)
            loss_ = beta * F.nll_loss(logits_single, labels[j].unsqueeze(0))
            loss += loss_
            logits_correction_total[j, :] = logits_correction
        logits_correction_total = logits_correction_total.cuda()
        loss = loss / len(labels)
        prec1,  = accuracy(logits_correction_total, labels, topk=(1, ))
        train_total+=1
        train_correct+=prec1
        loss.backward()
        optimizer.step()

        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f' 
                  %(epoch+1, args.n_epoch_2, i+1, len(train_dataset)//batch_size, prec1, loss.item()))
        
    train_acc=float(train_correct)/float(train_total)
    return train_acc

def val_correction(model, val_loader, epoch, W_group, basis_matrix_group, batch_size, num_classes, basis):
    print('Validating %s...' % model_str)
    
    val_total=0
    val_correct=0
   
    loss_total = 0.
    for i, (data, labels) in enumerate(val_loader):
        
        data = data.cuda()
        labels = labels.cuda()
        
        # Forward + Backward + Optimize
        loss = 0.
        _, logits=model(data, revision=False)
        
        logits_ = F.softmax(logits, dim=1)
        logits_correction_total = torch.zeros(len(labels), num_classes)
        for j in range(len(labels)):
            idx = i * batch_size + j
            matrix = matrix_combination(basis_matrix_group, W_group, idx, num_classes, basis)
            matrix = norm(matrix)
            matrix = torch.from_numpy(matrix).float().cuda()

            logits_single = logits_[j, :].unsqueeze(0)
            logits_correction = logits_single.mm(matrix)
            pro1 = logits_single[:, labels[j]]
            pro2 = logits_correction[:, labels[j]]
            beta = Variable(pro1/pro2, requires_grad=False)
            logits_correction = torch.log(logits_correction+1e-8)
            loss_ = beta * F.nll_loss(logits_correction, labels[j].unsqueeze(0))
            if torch.isnan(loss_) == True:
                loss_ = 0.
            loss += loss_
            logits_correction_total[j, :] = logits_correction

        logits_correction_total = logits_correction_total.cuda()
        loss = loss / len(labels)
        prec1,  = accuracy(logits_correction_total, labels, topk=(1, ))
        val_total+=1
        val_correct+=prec1

        loss_total += loss.item()

        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f' 
                  %(epoch+1, args.n_epoch_2, i+1, len(train_dataset)//batch_size, prec1, loss.item()))
        
    val_acc=float(val_correct)/float(val_total)

    return val_acc


def train_revision(model, train_loader, epoch, optimizer, W_group, basis_matrix_group, batch_size, num_classes, basis):
    print('Training %s...' % model_str)
    
    train_total=0
    train_correct=0 

    for i, (data, labels) in enumerate(train_loader):
        
        data = data.cuda()
        labels = labels.cuda()
        loss = 0.
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        _, logits, revision = model(data, revision=True)

        
        logits_ = F.softmax(logits, dim=1)
        logits_correction_total = torch.zeros(len(labels), num_classes)
        for j in range(len(labels)):
            idx = i * batch_size + j
            matrix = matrix_combination(basis_matrix_group, W_group, idx, num_classes, basis)
            matrix = torch.from_numpy(matrix).float().cuda()
            matrix = tools.norm(matrix + revision)
            
            logits_single = logits_[j, :].unsqueeze(0)
            logits_correction = logits_single.mm(matrix)
            pro1 = logits_single[:, labels[j]]
            pro2 = logits_correction[:, labels[j]]
            beta = pro1/ pro2
            logits_correction = torch.log(logits_correction+1e-12)
            logits_single = torch.log(logits_single+1e-12)
            loss_ = beta * F.nll_loss(logits_single, labels[j].unsqueeze(0))
            loss += loss_
            logits_correction_total[j, :] = logits_correction
        logits_correction_total = logits_correction_total.cuda()
        loss = loss / len(labels)
        prec1,  = accuracy(logits_correction_total, labels, topk=(1, ))
        train_total+=1
        train_correct+=prec1
        
        loss.backward()
        optimizer.step()

        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Train Accuracy: %.4F, Loss: %.4f' 
                  %(epoch+1, args.n_epoch_3, i+1, len(train_dataset)//batch_size, prec1, loss.item()))
        
    train_acc=float(train_correct)/float(train_total)
    return train_acc


def val_revision(model, train_loader, epoch, W_group, basis_matrix_group, batch_size, num_classes, basis):
   
    val_total=0
    val_correct=0

    for i, (data, labels) in enumerate(train_loader):
        model.eval()
        data = data.cuda()
        labels = labels.cuda()
        loss = 0.
        # Forward + Backward + Optimize
     
        _, logits, revision = model(data, revision=True)
        
        logits_ = F.softmax(logits, dim=1)
        logits_correction_total = torch.zeros(len(labels), num_classes)
        for j in range(len(labels)):
            idx = i * batch_size + j
            matrix = matrix_combination(basis_matrix_group, W_group, idx, num_classes, basis)
            matrix = torch.from_numpy(matrix).float().cuda()
            matrix = tools.norm(matrix + revision)
            logits_single = logits_[j, :].unsqueeze(0)
            logits_correction = logits_single.mm(matrix)
            pro1 = logits_single[:, labels[j]]
            pro2 = logits_correction[:, labels[j]]
            beta = Variable(pro1/pro2, requires_grad=True)
            logits_correction = torch.log(logits_correction+1e-12)
            loss_ = beta * F.nll_loss(logits_correction, labels[j].unsqueeze(0))
            loss += loss_
            logits_correction_total[j, :] = logits_correction
        logits_correction_total = logits_correction_total.cuda()
        prec1,  = accuracy(logits_correction_total, labels, topk=(1, ))
        val_total+=1
        val_correct+=prec1
        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Val Accuracy: %.4F, Loss: %.4f' 
                  %(epoch+1, args.n_epoch_3, i+1, len(val_dataset)//batch_size, prec1, loss.item()))
        
    val_acc = float(val_correct)/float(val_total)
   
    return val_acc






# Evaluate the Model
def evaluate(test_loader, model):
    print('Evaluating %s...' % model_str)
    model.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, labels in test_loader:
        
        data = data.cuda()
        _, logits = model(data, revision=False)
        outputs = F.softmax(logits, dim=1)
        _, pred1 = torch.max(outputs.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels.long()).sum()

    acc = 100*float(correct1)/float(total1)

    return acc


def respresentations_extract(train_loader, model, num_sample, dim_respresentations, batch_size):

    model.eval()
    A = torch.rand(num_sample, dim_respresentations)
    ind = int(num_sample / batch_size)
    with torch.no_grad():
        for i, (data, labels) in enumerate(train_loader):
            data = data.cuda()
            logits, _ = model(data, revision=False)
            if i < ind:
                A[i*batch_size:(i+1)*batch_size, :] = logits
            else:
                A[ind*batch_size:, :] = logits
      
    return A.cpu().numpy()


def probability_extract(train_loader, model, num_sample, num_classes, batch_size):

    model.eval()
    A = torch.rand(num_sample, num_classes)
    ind = int(num_sample / batch_size)
    with torch.no_grad():
        for i, (data, labels) in enumerate(train_loader):
            data = data.cuda()
            _ , logits = model(data, revision=False)
            logits = F.softmax(logits, dim=1)
            if i < ind:
                A[i*batch_size:(i+1)*batch_size, :] = logits
            else:
                A[ind*batch_size:, :] = logits
      
    return A.cpu().numpy()



def estimate_matrix(logits_matrix, model_save_dir):
    transition_matrix_group = np.empty((args.basis, args.num_classes, args.num_classes))
    idx_matrix_group = np.empty((args.num_classes, args.basis))
    a = np.linspace(97, 99, args.basis)
    a = list(a)
    for i in range(len(a)):
        percentage = a[i]
        index = int(i)
        logits_matrix_ = copy.deepcopy(logits_matrix)
        transition_matrix, idx = tools.fit(logits_matrix_, args.num_classes, percentage, True)
        transition_matrix = norm(transition_matrix)
        idx_matrix_group[:, index] = np.array(idx)
        transition_matrix_group[index] = transition_matrix
    idx_group_save_dir = model_save_dir + '/' + 'idx_group.npy'
    group_save_dir = model_save_dir + '/' + 'T_group.npy'
    np.save(idx_group_save_dir, idx_matrix_group) 
    np.save(group_save_dir, transition_matrix_group) 
    return idx_matrix_group, transition_matrix_group

def basis_matrix_optimize(model, optimizer, basis, num_classes, W_group, transition_matrix_group, idx_matrix_group, func, model_save_dir, epochs):
    basis_matrix_group = np.empty((basis, num_classes, num_classes))
    
    for i in range(num_classes):  

        model = tools.init_params(model)
        for epoch in range(epochs):
            loss_total = 0.
            for j in range(basis):
                class_1_idx = int(idx_matrix_group[i, j])
                W = list(np.array(W_group[class_1_idx, :]))
                T = torch.from_numpy(transition_matrix_group[j, i, :][:, np.newaxis]).float()
                prediction = model(W[0], num_classes)
                optimizer.zero_grad()
                loss = func(prediction, T)
                loss.backward()
                optimizer.step()
                loss_total += loss
            if loss_total < 0.02:
                break

        for x in range(basis):
            parameters = np.array(model.basis_matrix[x].weight.data)
    
            basis_matrix_group[x, i, :] = parameters
    A_save_dir = model_save_dir + '/' + 'A.npy'
    np.save(A_save_dir, basis_matrix_group)   
    return basis_matrix_group
    

def matrix_combination(basis_matrix_group, W_group, idx, num_classes, basis):
    coefficient = W_group[idx, :]

    M = np.zeros((num_classes, num_classes))
    for i in range(basis):
        
        temp = float(coefficient[0, i]) * basis_matrix_group[i, :, :]
        M += temp
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i,j]<1e-6:
                M[i,j] = 0.
    return M



def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               shuffle=False)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            num_workers=args.num_workers,
                                            drop_last=False,
                                            shuffle=False)



    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)
    # Define models
    print('building model...')
    if args.dataset == 'mnist':
        clf1 = LeNet()
    if args.dataset == 'fashionmnist':
        clf1 = resnet.ResNet18_F(10)
    if args.dataset == 'cifar10':
        clf1 = resnet.ResNet34(10)
    if args.dataset == 'svhn':
        clf1 = resnet.ResNet34(10)

    clf1.cuda()
    optimizer = torch.optim.SGD(clf1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    with open(txtfile, "a") as myfile:
        myfile.write('epoch train_acc val_acc test_acc\n')

    epoch = 0
    train_acc = 0
    val_acc = 0
    # evaluate models with random weights
    test_acc=evaluate(test_loader, clf1)
    print('Epoch [%d/%d] Test Accuracy on the %s test data: Model1 %.4f %%' % (epoch+1, args.n_epoch_1, len(test_dataset), test_acc))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' '  + str(train_acc)  + ' ' + str(val_acc) + ' ' + str(test_acc) + ' ' + "\n")


    best_acc = 0.0
    # training
    for epoch in range(1, args.n_epoch_1):
        # train models
        clf1.train()
        train_acc = train(clf1, train_loader, epoch, optimizer, nn.CrossEntropyLoss())
        # validation
        val_acc = evaluate(val_loader, clf1)
        # evaluate models
        test_acc = evaluate(test_loader, clf1)


        # save results
        print('Epoch [%d/%d] Train Accuracy on the %s train data: Model %.4f %%' % (epoch+1, args.n_epoch_1, len(train_dataset), train_acc))
        print('Epoch [%d/%d] Val Accuracy on the %s val data: Model %.4f %% ' % (epoch+1, args.n_epoch_1, len(val_dataset), val_acc))
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model %.4f %% ' % (epoch+1, args.n_epoch_1, len(test_dataset), test_acc))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' '  + str(train_acc) + ' ' + str(val_acc) + ' '  + str(test_acc) +  ' ' +  "\n")
            
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(clf1.state_dict(), model_save_dir + '/'+ 'model.pth')
            
    print('Matrix Factorization is doing...')    
    clf1.load_state_dict(torch.load(model_save_dir + '/'+ 'model.pth'))
    A = respresentations_extract(train_loader, clf1, len(train_dataset), args.dim, batch_size)
    A_val = respresentations_extract(val_loader, clf1, len(val_dataset), args.dim, batch_size)
    A_total = np.append(A, A_val, axis=0)
    W_total, H_total ,error= train_m(A_total, args.basis, args.iteration_nmf, 1e-5)
    for i in range(W_total.shape[0]):
        for j in range(W_total.shape[1]):
            if W_total[i,j]<1e-6:
                W_total[i,j] = 0.
    W = W_total[0:len(train_dataset), :]
    W_val = W_total[len(train_dataset):, :]
    print('Transition Matrix is estimating...Wating...')
    logits_matrix = probability_extract(train_loader, clf1, len(train_dataset), args.num_classes, batch_size)
    idx_matrix_group, transition_matrix_group = estimate_matrix(logits_matrix, model_save_dir)
    logits_matrix_val = probability_extract(val_loader, clf1, len(val_dataset), args.num_classes, batch_size)
    idx_matrix_group_val, transition_matrix_group_val = estimate_matrix(logits_matrix_val, model_save_dir)
    func = nn.MSELoss()

    model = Matrix_optimize(args.basis, args.num_classes)
    optimizer_1 = torch.optim.Adam(model.parameters(), lr=0.001)
    basis_matrix_group = basis_matrix_optimize(model, optimizer_1, args.basis, args.num_classes, W, 
                                               transition_matrix_group, idx_matrix_group, func, model_save_dir, args.n_epoch_4)
    
    basis_matrix_group_val = basis_matrix_optimize(model, optimizer_1, args.basis, args.num_classes, W_val, 
                                               transition_matrix_group_val, idx_matrix_group_val, func, model_save_dir, args.n_epoch_4)
    
    for i in range(basis_matrix_group.shape[0]):
        for j in range(basis_matrix_group.shape[1]):
            for k in range(basis_matrix_group.shape[2]):
                if basis_matrix_group[i, j, k] < 1e-6:
                    basis_matrix_group[i, j, k] = 0.

    optimizer_ = torch.optim.SGD(clf1.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)


    best_acc = 0.0
    for epoch in range(1, args.n_epoch_2):
        # train model
        clf1.train()
        
        train_acc = train_correction(clf1, train_loader, epoch, optimizer_, W, basis_matrix_group, batch_size, args.num_classes, args.basis)
        # validation
        val_acc = val_correction(clf1, val_loader, epoch, W_val, basis_matrix_group_val, batch_size, args.num_classes, args.basis)

        # evaluate models
        test_acc = evaluate(test_loader, clf1)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(clf1.state_dict(), model_save_dir + '/'+ 'model.pth')
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' '  + str(train_acc) + ' ' + str(val_acc) + ' '  + str(test_acc) +  ' ' +  "\n")
        # save results
        print('Epoch [%d/%d] Train Accuracy on the %s train data: Model %.4f %%' % (epoch+1, args.n_epoch_2, len(train_dataset), train_acc))
        print('Epoch [%d/%d] Val Accuracy on the %s val data: Model %.4f %% ' % (epoch+1, args.n_epoch_2, len(val_dataset), val_acc))
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model %.4f %% ' % (epoch+1, args.n_epoch_2, len(test_dataset), test_acc))
            
    clf1.load_state_dict(torch.load(model_save_dir + '/'+ 'model.pth'))
    optimizer_r = torch.optim.Adam(clf1.parameters(), lr=args.lr_revision, weight_decay=args.weight_decay)
    nn.init.constant_(clf1.T_revision.weight, 0.0)
    
    for epoch in range(1, args.n_epoch_3):
        # train models
        clf1.train()
        train_acc = train_revision(clf1, train_loader, epoch, optimizer_r, W, basis_matrix_group, batch_size, args.num_classes, args.basis)
        # validation
        val_acc = val_revision(clf1, val_loader, epoch, W_val, basis_matrix_group, batch_size, args.num_classes, args.basis)
        # evaluate models
        test_acc = evaluate(test_loader, clf1)
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' '  + str(train_acc) + ' ' + str(val_acc) + ' '  + str(test_acc) +  ' ' +  "\n")

        # save results
        print('Epoch [%d/%d] Train Accuracy on the %s train data: Model %.4f %%' % (epoch+1, args.n_epoch_3, len(train_dataset), train_acc))
        print('Epoch [%d/%d] Val Accuracy on the %s val data: Model %.4f %% ' % (epoch+1, args.n_epoch_3, len(val_dataset), val_acc))
        print('Epoch [%d/%d] Test Accuracy on the %s test data: Model %.4f %% ' % (epoch+1, args.n_epoch_3, len(test_dataset), test_acc))

if __name__=='__main__':
    main()
