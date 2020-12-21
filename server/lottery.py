import os
import logging
import numpy as np
import pickle
import random
import sys
import json
from multiprocessing import Process, Pool

import torch
import torch.multiprocessing as mp
from torchsummary import summary
import torchvision

from server import Server
from client.lth_client import LTHClient # pylint: disable=impoprt-error
import utils.dists as dists  # pylint: disable=no-name-in-module
import utils.fl_model as fl_model 
from utils.load_dataset import get_partition, get_train_set, get_testloader
import open_lth.models.registry as models_registry
from open_lth.cli import runner_registry
from open_lth.pruning.mask import Mask


class LotteryServer(Server):
    """server for open_lth"""

    def __init__(self, config):
        super().__init__(config)
    

    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server))

        self.static_global_model_path = self.config.paths.model
        
        # Add fl_model to import path
        sys.path.append(self.static_global_model_path)

        #get server split and clients total indices
        #server: server_indices, clients: label_idx_dict
        self.generate_dataset_splits()
        self.loading = self.config.data.loading
        if self.loading in ['static', 'dynamic_init']:
            self.get_clients_splits()

        # Set up simulated server
        self.init_model(self.static_global_model_path)
        self.make_clients()


    def init_model(self, static_global_model_path):

        lottery_runner = runner_registry.get(
            self.config.lottery_args.subcommand).create_from_args( \
                self.config.lottery_args)

        #set up global model
        self.model = models_registry.get(
            lottery_runner.desc.model_hparams, 
            outputs=lottery_runner.desc.train_outputs)

        self.save_model(self.model, static_global_model_path)
        self.baseline_weights = fl_model.extract_weights(self.model)

        #extract flattened weights
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, []) 

    #create clients without dataset assigned
    def make_clients(self):
        logging.info('Initializing clients...')
        clients = []
        
        for client_id in range(self.config.clients.total):
            
            new_client = LTHClient(client_id, self.config)
            clients.append(new_client)

        logging.info('Total clients: {}'.format(len(clients)))
            
        self.clients = clients

        logging.info('Download datasets if not exist...\n')
        warmup_client = LTHClient(-1, self.config)
        warmup_client.download_datasets()

    
    #get server_indices and label_idx_dict
    def generate_dataset_splits(self):
        
        dataset = get_train_set(self.config.lottery_args.dataset_name)

        self.labels = dataset.targets

        if torch.is_tensor(self.labels[0]):
            self.labels = [label.item() for label in self.labels]

        self.labels = list(set(self.labels))
               
        self.label_idx_dict = {}
        
        server_split = self.config.data.server_split
        self.server_indices = []
        #get label_idx_dict
        for label in self.labels:

            label_idx = self.get_indices(dataset, label)

            random.shuffle(label_idx)
            
            server_num = int(len(label_idx) * server_split)

            self.server_indices.extend(label_idx[:server_num])
            self.label_idx_dict[self.labels.index(label)] = \
                label_idx[server_num:]


    def get_clients_splits(self):
        tot_data_num = LotteryServer.get_dataset_num(self.config.lottery_args.dataset_name)
        if self.loading == 'static':
            client_num = self.config.clients.total
            tot_num = int(tot_data_num / client_num)
            overlap = False
        elif self.loading == 'dynamic_init':
            client_num = self.config.clients.total
            tot_num = self.config.data.partition['size']
            overlap = True
        else:
            client_num = self.config.clients.per_round
            tot_num = self.config.data.partition['size']
            overlap = True

        #get nums for each label for every client 
        client_idx_list = []
        for _ in range(client_num):
            client_idx = self.get_label_nums(tot_num)
            client_idx_list.append(client_idx)
        
        self.id_index_list = []
        
        for client_idx in client_idx_list:
            self.id_index_list.append(self.retrieve_indices(client_idx, overlap))
    

    #get indices list for a certain label distribution
    def retrieve_indices(self, client_idx, overlap):
        client_indices = []
        for label, idx in self.label_idx_dict.items():
            #already shuffle
            random.shuffle(idx)
            num = client_idx[self.labels.index(label)]
            client_indices.extend(idx[:num])
            if not overlap:
                #delete already retrieved indices
                idx = idx[num:]
        return client_indices
            

    def get_indices(self, dataset,label):
        indices =  []
        for i in range(len(dataset.targets)):
            if dataset.targets[i] == label:
                indices.append(i)
        return indices


    def get_label_nums(self, tot_num):
        #get number for each label list 
        client_idx = []
        if self.config.data.IID:
            for label, idx in self.label_idx_dict.items():
                client_idx.insert(self.labels.index(label), int(tot_num/len(self.labels)))
        
        else:
            bias = self.config.data.bias["primary"]
            secondary = self.config.data.bias["secondary"]
            
            pref = random.choice(self.labels)
            majority = int(tot_num * bias)
            minority = tot_num - majority
            #for one client get partition
            client_idx = get_partition(
                self.labels, majority, minority, pref, bias, secondary)

        return client_idx
        

    @staticmethod  
    def get_dataset_num(dataset_name):
        if dataset_name == "mnist":
            return 60000
        elif dataset_name == "cifar10":
            return 50000
        

    def round(self):
        sample_clients = self.selection()

        self.configuration(sample_clients)

        with Pool() as pool:
            processes = [pool.apply_async(client.run, ()) \
                for client in sample_clients]
            proc_results = [proc.get() for proc in processes]

        #get every client path
        sample_client_dict = {
            client.client_id: client for client in sample_clients}

        for client_id, data_folder, num_samples in proc_results:
            sample_client_dict[client_id].data_folder = data_folder
            sample_client_dict[client_id].report.set_num_samples(num_samples)

        self.testloader = get_testloader(
            self.config.lottery_args.dataset_name, self.server_indices) 
        
        return self.get_global_model(sample_clients)


    def get_global_model(self, sample_clients):

        reports = self.reporting(sample_clients)

        train_mode = self.config.lottery_args.subcommand

        if self.config.lottery_args.subcommand == "lottery":
            if self.config.fl.prune_level_setter == "greedy":
                return self.get_best_lottery(sample_clients, reports)
            elif self.config.fl.prune_level_setter == "rl-train":
                return self.train_best_model_rl(sample_clients, reports)
            elif self.config.fl.prune_level_setter == "rl-run":
                return self.get_best_model_rl(sample_clients, reports)
            elif self.config.fl.prune_level_setter == "manual":
                return self.get_pruned_model(
                    sample_clients, reports, self.config.lottery_args.levels)
            else:
                sys.exit("Configuration Error: lottery.subcommand, "\
                    +"federated_learning.prune_level_setter")

        elif self.config.lottery_args.subcommand == "train":
            return self.get_train_model(sample_clients, reports)
        else:
            sys.exit("Configuration Error: lottery.subcommand")


    def get_train_model(self, sample_clients, reports):
        client_paths = [client.data_folder for client in sample_clients]
        ep_num = int(self.config.lottery_args.training_steps[0:-2])

        weights = []
        
        for client_path in client_paths:
            weight_path = os.path.join(client_path, 'main', \
                f'model_ep{ep_num}_it0.pth')

            weight = self.get_model_weight(weight_path, True)
            weights.append(weight)

        updated_weights = self.federated_averaging(reports, weights)

        _, accuracy = self.test_model_accuracy(self.model, updated_weights)

        logging.info('Global accuracy: {:.2f}%\n'.format(100 * accuracy))

        # update static global model
        self.save_model(self.model, self.static_global_model_path)
        # backup global model to round directory
        # self.save_model(self.model, self.global_model_path_per_round)

        #get global model summary and sparsity report 
        fl_model.generate_sparsity_report(
            self.model, self.global_model_path_per_round +f'/sparsity_report.json')

        self.save_model_summary(
            self.model, self.global_model_path_per_round + f'/global_model.pth')
        
        with open(
            os.path.join(
                self.global_model_path_per_round, 'accuracy.json'), 'w') as fp:
            json.dump(accuracy, fp) 

        return accuracy


    def get_accuracy_per_level(self, sample_clients, reports, prune_level=None):

        client_paths = [client.data_folder for client in sample_clients]
        tot_level = self.config.lottery_args.levels + 1
        ep_num = int(self.config.lottery_args.training_steps[0:-2])

        accuracy_dict = {}
        #for loop
        for i in range(tot_level):      
            weights = []
            #load path to model 
            for client_path in client_paths:
                weight_path = os.path.join(client_path, f'level_{i}', 'main', 
                        f'model_ep{ep_num}_it0.pth')

                weight = self.get_model_weight(weight_path, True)
                weights.append(weight)

            #aggregation
            if self.config.fl.aggregation == "mask": 
                masks = []
                for client_path in client_paths:
                    mask_path = os.path.join(client_path, f'level_{i}', 'main', 
                        f'mask.pth')
                    mask = self.get_model_weight(mask_path, False)
                    masks.append(mask)
                
                updated_weights = self.federated_averaging_with_mask(
                    reports, weights, masks)
            
            if self.config.fl.aggregation == "normal":
                updated_weights = self.federated_averaging(reports, weights)

            #test accuracy 
            base_model = self.model
            base_model, accuracy_dict[i] = self.test_model_accuracy(
                base_model, updated_weights)
  
            model_path = os.path.join(self.global_model_path_per_round, \
                'global', f'level_{i}')

            if prune_level: # If specify level, only save this level model
                if prune_level == i: 
                    # backup global model of different levels to round directory
                    self.save_model(base_model, model_path, f'model.pth')
            else:
                # backup global model of different levels to round directory
                self.save_model(base_model, model_path, f'model.pth')

            #save model summary and sparsity report
            self.save_model_summary(
                base_model, model_path+f'/model.pth') 
            fl_model.generate_sparsity_report(
                base_model, model_path+f'/sparsity_report.json')      

        with open(os.path.join(
            self.global_model_path_per_round, 'accuracy.json'), 'w') as fp:
            json.dump(accuracy_dict, fp) 

        return accuracy_dict       


    def get_best_lottery(self, sample_clients, reports):

        accuracy_dict = self.get_accuracy_per_level(sample_clients, reports)       

        best_level = max(accuracy_dict, key=accuracy_dict.get)
        best_path = os.path.join(
            self.global_model_path_per_round, \
            'global', f'level_{best_level}', 'model.pth')

        #here self.model is the best model
        self.model.load_state_dict(torch.load(best_path))
        self.model.eval()

        # update static global model
        self.save_model(self.model, self.static_global_model_path)
        # backup the best global model to round directory
        # self.save_model(self.model, self.global_model_path_per_round)

        accuracy = accuracy_dict[best_level]
        logging.info(f'Level {best_level} Model '\
            +'Best average accuracy: {:.2f}%\n'.format(100 * accuracy))

        return accuracy


    def save_global_mask(self, model,path):
        # Get the model weights.
        prunable_layers = []
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.conv.Conv2d) \
                or isinstance(module, torch.nn.modules.linear.Linear):
            
                prunable_layers.append(name+'weight')
            
        prunable_tensors = set(prunable_layers)
                
        weights = {k: v.clone().cpu().detach().numpy()
                   for k, v in model.state_dict().items()
                   if k in prunable_tensors}
        for k, v in model.state_dict().items():
            if k in prunable_tensors:
                weights[k] = v.clone().cpu().detach().numpy()
    
        new_mask = Mask({k: self.div0(v,v)
                         for k, v in weights.items()})
        
        #bias part
        current_mask = Mask.ones_like(model).numpy() 
        for k in current_mask:
            if k not in new_mask:
                new_mask[k] = current_mask[k]
        
        torch.save({k: v.cpu().int() for k, v in new_mask.items()}, path)

        
    def get_pruned_model(self, sample_clients, reports, prune_level):
        # accuracy_dict = { level: global_model_accuracy }
        accuracy_dict = self.get_accuracy_per_level(
            sample_clients, reports, prune_level)

        selected_model_path = os.path.join(self.global_model_path_per_round, \
            'global', f'level_{prune_level}', 'model.pth')
        
        self.model.load_state_dict(torch.load(selected_model_path))
        self.model.eval()

        #get best global model mask and save to the static mask path(update with every round)
        #self.save_global_mask(
            # self.model, self.static_global_model_path+f'/mask.pth')

        # update static global model for next round
        self.save_model(self.model, self.static_global_model_path)
        # backup the seleted global model to round directory
        # self.save_model(self.model, self.global_model_path_per_round)

        accuracy = accuracy_dict[prune_level]
        logging.info(f'Selected level-{prune_level} model accuracy: '\
            + '{:.2f}%'.format(100 * accuracy))
        
        return accuracy
          

    def train_best_model_rl(self, sample_clients, reports):
        pass
    

    def get_best_model_rl(self, sample_clients, reports):
        pass


    def test_model_accuracy(self, model, updated_weights):
        fl_model.load_weights(model, updated_weights)
        accuracy = fl_model.test(model, self.testloader)

        return model, accuracy 


    def get_model_weight(self, path, strict):

        model = self.model
        model.load_state_dict(torch.load(path), strict=strict)
        model.eval()
        #the weight pth not mask pth
        if strict:
            self.save_model_summary(model, path)
        #weight: tensor
        weight = fl_model.extract_weights(model)
        return weight


    def save_model_summary(self, model, pth_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset_name = self.config.lottery_args.dataset_name.strip()
        if dataset_name == 'mnist':
            size = (1, 28, 28)
        elif dataset_name == 'cifar10':
            size = (3, 32, 32)
        elif dataset_name == 'fashion_mnist':
            size = (1, 28, 28)
        else:
            print("dataset name is wrong!")

        fpath = pth_path[:-4]+'summary'

        if not os.path.exists(fpath[:-9]):
            os.makedirs(fpath[:-9])

        stdoutOrigin = sys.stdout
        sys.stdout = open(fpath, 'w+')
        summary(model.to(device), size)
        sys.stdout.close()
        sys.stdout=stdoutOrigin


    def configuration(self, sample_clients):

        display = self.config.clients.display_data_distribution
        for i in range(len(sample_clients)):   
            client = sample_clients[i]
            
            if self.loading in ['static', 'dynamic_init']:
                dataset_indices = self.id_index_list[client.client_id]            
            elif self.loading == 'dynamic':
                self.get_clients_splits()
                dataset_indices = self.id_index_list[i]
            else:
                dataset_indices = self.id_index_list[client.client_id] 

            client.set_data_indices(dataset_indices)
            
            if display:
                self.display_data_distribution(client)

    
    def display_data_distribution(self, client):
        
        dataset_indices = client.dataset_indices

        label_cnt = []
        for i in range(len(self.labels)):
            tot_idx = set(self.label_idx_dict[i])
            intersection = tot_idx.intersection(dataset_indices)
            label_cnt.append(len(intersection))

        tot_num = sum(label_cnt)
        
        if self.config.data.IID:
            logging.info(
                f'Total {tot_num} data in client {client.client_id}, {label_cnt[0]} for one label.')

        else:
            pref_num = max(label_cnt)
            bias = round(pref_num / tot_num,2)
            logging.info(
                f'Total {tot_num} data in client {client.client_id},\
                    label {label_cnt.index(pref_num)} has {bias} of total data.')
            

    def get_total_sample(self, masks):

        total_num = [torch.zeros(x.size()) for _,x in masks[0]]
        for mask in masks:
            for i, (name, mask) in enumerate(mask):
                print(name)
                print(mask)
                total_num[i]+= mask

        return total_num


    def federated_averaging_with_mask(self, reports, weights, masks):
        
        # Extract updates from reports
        updates = self.extract_client_updates(weights)

        # Extract layer mask 
        tot_mask = self.get_total_sample(masks)
        tot_samples = sum([report.num_samples for report in reports])
        
        avg_update = [torch.zeros(x.shape)  # pylint: disable=no-member
                      for _, x in updates[0]]
        
        mask_for_updates = [np.ones(x.shape)  # pylint: disable=no-member
                      for _, x in updates[0]]

        
        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples
            for j, (name, delta) in enumerate(update):
                if 'bias' in name:
                    avg_update[j] += delta * (num_samples/tot_samples)
                    
                else:
                    num = tot_mask[j].numpy() 
                    delta = delta.numpy()
                    avg_update[j] += torch.from_numpy(self.div0(delta, num))
                    #get 0 for num=0
                    mask_for_updates[j] = self.div0(num, num)
        

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(self.baseline_weights):
            weight = torch.from_numpy(np.multiply(weight.numpy(), 
                                                    mask_for_updates[i]))
            
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights


    def div0(self, a,b):
   
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide( a, b )
            c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
        return c
