# PyTorch
from torchvision import transforms, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision.datasets import ImageFolder

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd
import os
import shutil

# Image manipulations
from PIL import Image
# Useful for examining network
from torchsummary import summary
# Timing utility
from timeit import default_timer as timer


def device_to_use():
    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')

    #If you use multiple gpus it turns statement multi_gpu = True. Probably useful for large datasets
    # Number of gpus
    multi_gpu = False
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False
    return train_on_gpu, multi_gpu


def split_train(train_data_folder, train_fraction,transform):
    # print(train_data_folder)
    data_train = ImageFolder(train_data_folder, transform)
    train, test = random_split(data_train, [int(train_fraction*len(data_train)),len(data_train)-int(train_fraction*len(data_train))], generator=torch.Generator().manual_seed(42))
    return train, test


def prepare_data_classes_from_train(src_data_folder, dest_data_folder, train_labels_csv_file, move = False):
    data = pd.read_csv(train_labels_csv_file)
    labels_dic = {}
    for id, pos_label in zip(data['id'], data['pos_label']):
        labels_dic[str(id).split(".")[0]] = int(pos_label)
    # print(data.groupby('pos_label').count())
    n_classes = data.groupby('pos_label').count().count()[0]

    if os.path.exists(f'{dest_data_folder}'):
        shutil.rmtree(f'{dest_data_folder}')
    os.mkdir(f'{dest_data_folder}')


    for cls in range(n_classes):
        if os.path.exists(f'{dest_data_folder}/{cls}'):
            shutil.rmtree(f'{dest_data_folder}/{cls}')
        os.mkdir(f'{dest_data_folder}/{cls}')

    for file in os.listdir(src_data_folder):
        # print(labels_dic[file.split('.')[0]])
        # print(f'{dest_data_folder}/{str(labels_dic[file.split(".")[0]])}/{file}')

        try:            
            # print(f'{src_data_folder}/{file} ==> {dest_data_folder}/{str(labels_dic[file.split(".")[0]])}/{file}')
            if move:
                shutil.move(f'{src_data_folder}/{file}', f'{dest_data_folder}/{str(labels_dic[file.split(".")[0]])}/{file}')
            else:
                shutil.copy(f'{src_data_folder}/{file}', f'{dest_data_folder}/{str(labels_dic[file.split(".")[0]])}/{file}')
        except Exception:
            print(f'Error in copying ... file: {file}')

    return n_classes

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def train(model,criterion,optimizer,train_loader, test_loader = None ,n_epochs=1, print_log = True):
    history = []
    n_batches = len(train_loader)
    # Main loop
    if print_log:
        print(f'running train loop for {n_epochs} epochs ...')
    for epoch in range(n_epochs):
        n_corrects = 0

        train_batch_loss_history = []
        train_batch_acc_history = []
        train_loss = 0.0
        # valid_loss = 0.0

        train_acc = 0
        # valid_acc = 0

        batch_size = train_loader.batch_size
        # Set to training
        model.train()
        start = timer()

        # with tqdm(total=n_batches) as pbar:
        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # pbar.update(1)
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)
            
            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            n_batch_correct = torch.sum(pred == target.data)
            data_size_in_batch = len(target.data)
            n_corrects += n_batch_correct

            # correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            # batch_accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            batch_accuracy = n_batch_correct/data_size_in_batch
            train_acc += batch_accuracy.item() * data.size(0)


            # Track training progress
            
            # print(
            #     f'Batch {ii} in epoch {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
            #     end='\r')
            
            # printProgressBar(ii + 1, len(train_loader), prefix = f'Epoch {epoch}: ', suffix = f'{timer() - start:.2f} sec. Acc = {batch_accuracy*100:.2f}, acc ={n_corrects*100/(data_size_in_batch+ii*batch_size):.2f}', length = n_batches)
            if print_log:
                printProgressBar(ii + 1, len(train_loader), prefix = f'Epoch {epoch}: ', suffix = f'{timer() - start:.2f} sec. Batch acc: {batch_accuracy*100:.2f}%, loss: {loss.item():.4f}', length = 50)            
            train_batch_loss_history.append(loss.item())
            train_batch_acc_history.append(batch_accuracy.item())

        epoch_loss = train_loss / len(train_loader.sampler)
        epoch_acc = n_corrects / len(train_loader.sampler)# len(train_loader.sampler) = train_data_loader size
        epoch_avg_acc = train_acc / len(train_loader.sampler)
        if print_log:
            print(f'Loss: {epoch_loss*100:.4f}% Acc: {epoch_acc*100:.4f}%')

        acc_test_train = -1
        if test_loader:
            acc_test_train,_,_  = test(model, test_loader, print_log)
            if print_log:
                print(f'Test set Acc: {acc_test_train*100:.4f}%')

        history.append([acc_test_train, epoch_loss, epoch_acc.item(), train_batch_loss_history, train_batch_acc_history])
    return model, history


def test(model,test_loader, print_log = True):
    history = []
    n_corrects = 0
    test_loss = 0.0
    y_pred = []
    y_true = []

    n_batches = len(test_loader)
    # Don't need to keep track of gradients
    with torch.no_grad():
        model.eval()
        # start = timer()
        # ii = 1
        for data, target in test_loader:
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Forward pass
            output = model(data)

            # Calculate loss
            # loss = criterion(output, target)
            # Multiply average loss times the number of examples in batch
            # test_loss += loss.item() * data.size(0)

            # Calculate accuracy
            _, pred = torch.max(output, dim=1)
            n_corrects += torch.sum(pred == target.data)

            y_pred.append(pred)
            y_true.append(target.data)


            # batch_accuracy = n_corrects / data.size(0)
            # printProgressBar(ii , len(test_loader), prefix = f'Batch {ii}: ', suffix = f'{timer() - start:.2f} sec. Batch acc: {batch_accuracy*100:.2f}', length = n_batches)            
            # ii+=1
        # Calculate average losses
        # test_loss = test_loss / len(test_loader.sampler)

        test_acc = n_corrects / len(test_loader.sampler)

        history.append([ test_acc.item()])

        y_pred_ = torch.cat(y_pred, dim=0) 
        y_true_ = torch.cat(y_true, dim=0) 
        if print_log:
            print(f'Accuracy: {test_acc*100:.4f}%')
    return test_acc.item(), y_pred_, y_true_


def get_pretrained_model(model_name, n_classes):

    
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[-1].in_features

        # Add on classifier
        model.classifier[-1] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

        # initialize the weights
        # model.classifier[-2].apply(init_weights)
        model.classifier[-1].apply(init_weights)


    if model_name == 'simple':
        model =  nn.Sequential(
            nn.Linear(1000, 800), nn.ReLU(), 
            nn.Linear(800, 500), nn.ReLU(),  
            nn.Linear(500, 250), nn.ReLU(),  nn.Dropout(0.2),
            nn.Linear(250, 100), nn.ReLU(),  nn.Dropout(0.2),
            nn.Linear(100, n_classes), nn.LogSoftmax(dim=1))

        # initialize the weights
        # model.classifier[-2].apply(init_weights)
        model.apply(init_weights)


    # Move to gpu and parallelize-
    if train_on_gpu:
        model = model.to('cuda')

    return model    

def init_weights(m):
    if isinstance(m, nn.Linear):
        if train_on_gpu:
            torch.cuda.manual_seed(2023)
        else:
            torch.manual_seed(2023)
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


train_on_gpu, multi_gpu = device_to_use()

def get_csv_test_labels(csv_filename, model, test_files, test_loader):
    with torch.no_grad():
        model.eval()
        for data, target in test_loader:
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Forward pass
            output = model(data)

            
            with open(csv_filename, 'w') as f:
                for d, t in zip(data, target):
                    f.write(str(d))
                    f.write(str(t))


#     # Calculate loss
#     loss = criterion(output, target)
#     # Multiply average loss times the number of examples in batch
#     # test_loss += loss.item() * data.size(0)

#     # Calculate accuracy
#     _, pred = torch.max(output, dim=1)
#     n_corrects += torch.sum(pred == target.data)
#     # batch_accuracy = n_corrects / data.size(0)
#     # printProgressBar(ii , len(test_loader), prefix = f'Batch {ii}: ', suffix = f'{timer() - start:.2f} sec. Batch acc: {batch_accuracy*100:.2f}', length = n_batches)            
#     # ii+=1
# # Calculate average losses
# # test_loss = test_loss / len(test_loader.sampler)

# test_acc = n_corrects / len(test_loader.sampler)

# history.append([ test_acc.item()])


# print(f'Acc: {test_acc*100:.4f}%')
    # return test_acc.item()
