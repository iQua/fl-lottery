import client
import os
import logging
import numpy as np
import pickle
import random
import sys
import shutil

import torch
import utils.dists as dists  # pylint: disable=no-name-in-module
from utils.fl_model import load_weights, extract_weights # pylint: disable=no-name-in-module
from datetime import datetime
import pytz

def current_time():
    tz_NY = pytz.timezone('America/New_York') 
    datetime_NY = datetime.now(tz_NY)
    return datetime_NY.strftime("%m_%d_%H:%M:%S")


class Server(object):
    """Basic federated learning server."""

    def __init__(self, config, env=None, agent=None):
        self.config = config
        self.saved_reports = {}
        
        self.agent = agent
        self.env = env

        self.init_run_path()
        
        # Set logging
        logging.basicConfig(
            filename=os.path.join(self.current_run_path, 'logger.log'), 
            format='[%(threadName)s][%(asctime)s]: %(message)s', 
            level=self.config.log_level, 
            datefmt='%H:%M:%S')


    def init_run_path(self):
        
        if self.config.lottery_args.subcommand == "lottery":
            self.current_run_path = os.path.join("/mnt/open_lth_data",\
                current_time()+"-"+self.config.lottery_args.subcommand\
                    +"_"+self.config.fl.prune_level_setter)
        else:
            self.current_run_path = os.path.join("/mnt/open_lth_data",\
                current_time()+"-"+self.config.lottery_args.subcommand)

        if not os.path.exists(self.current_run_path):
            os.mkdir(self.current_run_path)


    # Set up server
    def boot(self):
        pass


    def load_model(self):
        pass


    def make_clients(self, num_clients):
        pass


    # Run federated learning
    def run(self):
        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        # Perform rounds of federated learning
        for round_id in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round_id, rounds))

            self.set_params(round_id)
        
            # Run the federated learning round
            accuracy = self.round()

            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break

        if reports_path:
            with open(reports_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_path))


    def set_params(self, round_id):   
        self.config.lottery_args.round_num = round_id        
        self.config.lottery_args.client_num = self.config.clients.total

        self.global_model_path_per_round = os.path.join(
            self.current_run_path, str(round_id))

        self.config.lottery_args.prefix_path = self.current_run_path

        # Static global model path
        self.config.lottery_args.global_model_path = os.path.join(
            self.config.paths.model)

        # Backup config file
        shutil.copyfile(self.config.config_path, \
            os.path.join(self.current_run_path, "config.json"))
 

    def round(self):
        return 0

    # Federated learning phases

    def selection(self):
        # Select devices to participate in round
        clients_per_round = self.config.clients.per_round

        # Select clients randomly
        sample_clients = [client for client in random.sample(
            self.clients, clients_per_round)]

        return sample_clients


    def configuration(self, sample_clients):
        pass


    def reporting(self, sample_clients):
        # Recieve reports from sample clients
        reports = [client.get_report() for client in sample_clients]

        logging.info('Reports recieved: {}'.format(len(reports)))
        assert len(reports) == len(sample_clients)

        return reports


    # Report aggregation
    def extract_client_updates(self, weights):
        # Calculate updates from weights
        updates = []
        for weight in weights:
            update = []
            for i, (name, weight) in enumerate(weight):
                bl_name, baseline = self.baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate update
                delta = weight - baseline
                update.append((name, delta))
            updates.append(update)

        return updates


    def federated_averaging(self, reports, weights):
        
        # Extract updates from reports
        updates = self.extract_client_updates(weights)

        # Extract total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        avg_update = [torch.zeros(x.size())  # pylint: disable=no-member
                      for _, x in updates[0]]
        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples
            for j, (_, delta) in enumerate(update):
                # Use weighted average by number of samples
                avg_update[j] += delta * (num_samples / total_samples)


        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(self.baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights


    # Server operations
    @staticmethod
    def flatten_weights(weights):
        # Flatten weights into vectors
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)


    def save_model(self, model, path, filename=None):
        if not os.path.exists(path):
            os.makedirs(path)
        
        if filename:
            path += '/'+filename
        else:
            path += '/global.pth'
        
        torch.save(model.state_dict(), path)
        logging.info('Saved global model: {}'.format(path))


    def save_reports(self, round, reports):

        if reports:
            self.saved_reports['round{}'.format(round)] = [(
                report.client_id, self.flatten_weights(report.weights)) 
                for report in reports]

        # Extract global weights
        self.saved_reports['w{}'.format(round)] = self.flatten_weights(
            extract_weights(self.model))
