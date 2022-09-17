# Load libraries
import math, random, copy, os, glob, time
from itertools import chain, combinations, permutations
from pprint import pprint

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision import datasets, transforms as T

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from utils.parameters import args_parser
from utils.data_process import prepareIID, prepareNIID1, prepareNIID2, prepareNIID12
from models.Nets import SmallMLP_MNIST, MediumMLP_MNIST, LargeMLP_MNIST, SmallMLP_EMNIST, MediumMLP_EMNIST, LargeMLP_EMNIST
from utils.save_file import createDirectory, deleteAllModels, saveCheckpoint, print_parameters, loadCheckpoint
from models.Fed import FedAvg
from utils.helpers import powerset, grangerset, aggListOfDicts, getAllClients

from models.server import server
from models.clients import initClients

def train(dataloader, model, loss_fn, optimizer, verbose=False):
    '''
        Trains a NN model over a dataloader
    '''
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)

            if verbose:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            return loss


def test(dataloader, model, loss_fn, verbose=False):
    '''
        Tests a NN model over a dataloader
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct, f1 = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            f1 += f1_score(y.cpu(), y_pred.argmax(1).cpu(), average='micro')

    test_loss /= num_batches
    correct /= size
    f1 /= num_batches

    if verbose:
        print(f"Test Error: \n Accuracy: {correct:>8f}, Avg loss: {test_loss:>8f}, F1: {f1:>8f} \n")

    return test_loss, correct, f1


def trainClients(clients, server):
    '''
        Trains a list of client devices and saves their parameters
    '''
    loss, acc, f1 = {}, {}, {}
    for client in clients:
        train_loss, test_loss, test_acc, test_f1 = trainClient(client, server)

        # Aggregate statistics
        loss[client['name']] = test_loss
        acc[client['name']] = test_acc
        f1[client['name']] = test_f1

    return loss, acc, f1


def trainClient(client, server):
    '''
        Trains a client device and saves its parameters
    '''
    # Read client behaviour setting
    client_behaviour = client['behaviour']

    # Load local dataset
    client_dataloader = client['dataloader']

    # Get client model and functions
    client_name = client['name']

    client_model = FederatedModel().to(device)
    client_loss_fn = FederatedLossFunc()
    client_optimizer = FederatedOptimizer(client_model.parameters(), lr=FederatedLearnRate, momentum=FederatedMomentum,
                                          weight_decay=FederatedWeightDecay)

    # If client is adversarial, they return randomized parameters
    if client_behaviour == 'ADVERSARIAL':
        # Save client model state_dicts (simulating client uploading model parameters to server)
        saveCheckpoint(
            client_name,
            client_model.state_dict(),
            client_optimizer.state_dict(),
            client['filepath'],
        )

        test_loss, test_acc, test_f1 = test(server['dataloader'], client_model, client_loss_fn)
        print(f"{client_name} ({client_behaviour}) Test Acc: {test_acc:>8f}, Loss: {test_loss:>8f}, F1: {test_f1:>8f}")

        return 0, test_loss, test_acc, test_f1

    # Load server model state_dicts (simulating client downloading server model parameters)
    checkpoint = loadCheckpoint(server['filepath'])
    client_model.load_state_dict(checkpoint['model_state_dict'])  # Using current server model parameters
    # client_optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Using current server model parameters

    # If client is a freeloader, they return the same server model parameters
    if client_behaviour == 'FREERIDER':
        # Save client model state_dicts (simulating client uploading model parameters to server)
        saveCheckpoint(
            client_name,
            client_model.state_dict(),
            client_optimizer.state_dict(),
            client['filepath'],
        )

        test_loss, test_acc, test_f1 = test(server['dataloader'], client_model, client_loss_fn)
        print(f"{client_name} ({client_behaviour}) Test Acc: {test_acc:>8f}, Loss: {test_loss:>8f}, F1: {test_f1:>8f}")

        return 0, test_loss, test_acc, test_f1

    # If client is normal, they train client over N epochs
    epochs = args.epoch
    print(f'Training {client_name} over {epochs} epochs...')
    for t in range(epochs):
        train_loss = train(client_dataloader, client_model, client_loss_fn, client_optimizer)

    test_loss, test_acc, test_f1 = test(server['dataloader'], client_model, client_loss_fn)
    print(f"{client_name} ({client_behaviour}) Test Acc: {test_acc:>8f}, Loss: {test_loss:>8f}, F1: {test_f1:>8f}")

    # Save client model state_dicts (simulating client uploading model parameters to server)
    saveCheckpoint(
        client_name,
        client_model.state_dict(),
        client_optimizer.state_dict(),
        client['filepath'],
    )

    return train_loss, test_loss, test_acc, test_f1


def evalFedAvg(server):
    '''
        Load client state dicts, perform parameter aggregation and evaluate contributions for each client
    '''
    # Retrieve all clients' uploaded data
    client_filepaths = glob.glob(f"{server['client_filepath']}/client*.pt")

    # Load client model state_dicts (simulating client downloading server model parameters)
    client_checkpoints = []
    for client_filepath in client_filepaths:
        client_checkpoint = loadCheckpoint(client_filepath)
        client_checkpoints += [client_checkpoint]

    # Get Federated Average of clients' parameters
    model_state_dicts = [checkpoint['model_state_dict'] for checkpoint in client_checkpoints]
    fed_model_state_dict = FedAvg(model_state_dicts)

    # Instantiate server model using FedAvg
    fed_model = FederatedModel().to(device)
    fed_model.load_state_dict(fed_model_state_dict)
    fed_model.eval()

    # Evaluate FedAvg server model
    start_time = time.time()  # Time evaluation period
    eval_loss, eval_acc, eval_f1 = test(server['dataloader'], fed_model, server['loss_func'])
    time_taken = time.time() - start_time  # Get model evaluation period (in seconds)
    print(f"\n>> Federated Model Acc: {eval_acc:>8f}, Loss: {eval_loss:>8f}, F1: {eval_f1:>8f}\n")

    # Save server model state_dicts (simulating public access to server model parameters)
    saveCheckpoint(
        server['name'],
        fed_model.state_dict(),
        server['optimizer'],
        server['filepath'],
    )

    # Output statistics
    return eval_loss, eval_acc, eval_f1, time_taken


def trainFedAvgModel(rounds):
    '''
        Train a model using naive FedAvg
    '''

    loss, acc, f1, eval_time = [], [], [], []
    for i in range(rounds):
        print(f'\n=======================\n\tROUND {i + 1}\n=======================')
        clients_loss, clients_acc, clients_f1 = trainClients(clients, server)
        server_loss, server_acc, server_f1, time_taken = evalFedAvg(server)

        # Compile performance measures
        loss += [{**clients_loss, **{'server': server_loss}}]
        acc += [{**clients_acc, **{'server': server_acc}}]
        f1 += [{**clients_f1, **{'server': server_f1}}]
        eval_time += [time_taken]

    # Output statistics
    return aggListOfDicts(loss), aggListOfDicts(acc), aggListOfDicts(f1), eval_time


if __name__ == '__main__':
    # parse args
    args = args_parser()

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create subdirectories
    model_path = './models'
    # Initialize system and define helper functions
    createDirectory(model_path)
    # Delete existing .pt files from previous run
    deleteAllModels(model_path)

    # Download training and test data from open datasets
    # MLP model uses Fashion-MNIST
    if args.dataset_type == 'MNIST':
        train_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=T.ToTensor(),
        )

        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=T.ToTensor(),
        )

    elif args.dataset_type == 'EMNIST':
        train_data = datasets.EMNIST(
            root="data",
            train=True,
            download=True,
            split='balanced',
            transform=T.ToTensor(),
        )

        test_data = datasets.EMNIST(
            root="data",
            train=False,
            download=True,
            split='balanced',
            transform=T.ToTensor(),
        )

    # Split training dataset for clients
    NUM_OF_CLIENTS = args.num_normal_clients + args.num_freerider_clients + args.num_adversarial_clients
    if args.distribution_type == 'IID':
        train_datasets = prepareIID(train_data, NUM_OF_CLIENTS)
    elif args.distribution_type == 'NIID_1':
        train_datasets = prepareNIID1(train_data, NUM_OF_CLIENTS)
    elif args.distribution_type == 'NIID_2':
        train_datasets = prepareNIID2(train_data, NUM_OF_CLIENTS)
    elif args.distribution_type == 'NIID_12':
        train_datasets = prepareNIID12(train_data, NUM_OF_CLIENTS)

    train_dataloaders = [DataLoader(train_dataset, batch_size=args.batchsize) for train_dataset in train_datasets]
    test_dataloader = DataLoader(test_data, batch_size=args.batchsize * 2)

    # Define network model architecture
    FederatedModel = None

    if (args.model_size == 'SMALL') & (args.dataset_type == 'MNIST'):
        FederatedModel = SmallMLP_MNIST
    elif (args.model_size == 'MEDIUM') & (args.dataset_type == 'MNIST'):
        FederatedModel = MediumMLP_MNIST
    elif (args.model_size == 'LARGE') & (args.dataset_type == 'MNIST'):
        FederatedModel = LargeMLP_MNIST
    elif (args.model_size == 'SAMLL') & (args.dataset_type == 'EMNIST'):
        FederatedModel = SmallMLP_EMNIST
    elif (args.model_size == 'MEDIUM') & (args.dataset_type == 'EMNIST'):
        FederatedModel = MediumMLP_MNIST
    elif (args.model_size == 'LARGE') & (args.dataset_type == 'EMNIST'):
        FederatedModel = LargeMLP_EMNIST

    # Define network training functions and hyper-parameters
    # Training hyper-parameters and functions for the Federated modeel
    FederatedLossFunc = nn.CrossEntropyLoss
    FederatedOptimizer = torch.optim.SGD
    FederatedLearnRate = args.learning_rate
    FederatedMomentum = args.momentum
    FederatedWeightDecay = args.weight_decay


    # Initalize server and clients
    server = server(FederatedModel, FederatedLossFunc, FederatedOptimizer, FederatedLearnRate, FederatedMomentum,
                        FederatedWeightDecay)
    server = server.initServer(model_path, 'FedAvg', test_dataloader)
    clients = initClients(args.num_normal_clients, args.num_freerider_clients, args.num_adversarial_clients, server,
                              train_dataloaders)

    # Train and evaluate
    fedavg_loss, fedavg_acc, fedavg_f1, fedavg_time = trainFedAvgModel(args.common_rounds)