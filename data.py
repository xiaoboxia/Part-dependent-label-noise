import numpy as np
import torch.utils.data as Data
from PIL import Image

import tools
import torch

class mnist_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.2, split_percentage=0.9, seed=1, num_classes=10, feature_size=28*28, norm_std=0.1):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        original_images = np.load('data/mnist/train_images.npy')
        original_labels = np.load('data/mnist/train_labels.npy')
        data = torch.from_numpy(original_images).float()
        targets = torch.from_numpy(original_labels)

        dataset = zip(data, targets)
        new_labels = tools.get_instance_noisy_label(noise_rate, dataset, targets, num_classes, feature_size, norm_std, seed)
       	
        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.data_split(original_images, new_labels, split_percentage,seed)

    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
   
        else:
            return len(self.val_data)
 

class mnist_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
        
        self.test_data = np.load('data/mnist/test_images.npy')
        self.test_labels = np.load('data/mnist/test_labels.npy') - 1 # 0-9
        
    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    
    def __len__(self):
        return len(self.test_data)
    
class cifar10_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.2, split_percentage=0.9, seed=1, num_classes=10, feature_size=3*32*32, norm_std=0.1):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        
        original_images = np.load('data/cifar10/train_images.npy')
        original_labels = np.load('data/cifar10/train_labels.npy')
        data = torch.from_numpy(original_images).float()
        targets = torch.from_numpy(original_labels)

        dataset = zip(data, targets)
        new_labels = tools.get_instance_noisy_label(noise_rate, dataset, targets, num_classes, feature_size, norm_std, seed)


        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.data_split(original_images, new_labels, split_percentage,seed)
        if self.train:      
            self.train_data = self.train_data.reshape((-1,3,32,32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) 
            print(self.train_data.shape)
        
        else:
            self.val_data = self.val_data.reshape((-1, 3,32,32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)
        
class cifar10_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load('data/cifar10/test_images.npy')
        self.test_labels = np.load('data/cifar10/test_labels.npy')
        self.test_data = self.test_data.reshape((10000,3,32,32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1)) 

    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    
    def __len__(self):
        return len(self.test_data)
    
class svhn_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.2, split_percentage=0.9, seed=1, num_classes=10, feature_size=3*32*32, norm_std=0.1):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        
        original_images = np.load('data/svhn/train_images.npy')
        original_labels = np.load('data/svhn/train_labels.npy')
        data = torch.from_numpy(original_images).float()
        targets = torch.from_numpy(original_labels)

        dataset = zip(data, targets)
        new_labels = tools.get_instance_noisy_label(noise_rate, dataset, targets, num_classes, feature_size, norm_std, seed)

        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.data_split(original_images, new_labels, split_percentage,seed)
        if self.train:      
            self.train_data = self.train_data.reshape((-1,3,32,32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) 
        
        else:
            self.val_data = self.val_data.reshape((-1, 3,32,32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)
        
class svhn_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load('data/svhn/test_images.npy')
        self.test_labels = np.load('data/svhn/test_labels.npy')
        self.test_data = self.test_data.reshape((-1,3,32,32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1)) 

    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    
    def __len__(self):
        return len(self.test_data)
    
class fashionmnist_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, noise_rate=0.2, split_percentage=0.9, seed=1, num_classes=10, feature_size=784, norm_std=0.1):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        
        original_images = np.load('data/fashionmnist/train_images.npy')
        original_labels = np.load('data/fashionmnist/train_labels.npy')
        data = torch.from_numpy(original_images).float()
        targets = torch.from_numpy(original_labels)

        dataset = zip(data, targets)
        new_labels = tools.get_instance_noisy_label(noise_rate, dataset, targets, num_classes, feature_size, norm_std, seed)
       

        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.data_split(original_images, new_labels, split_percentage,seed)

    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)
        
class fashionmnist_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load('data/fashionmnist/test_images.npy')
        self.test_labels = np.load('data/fashionmnist/test_labels.npy')


    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    
    def __len__(self):
        return len(self.test_data)
    

