import argparse
import logging
import os
import sys
import pytz
from datetime import datetime
import copy

import client
import config
import server
from rl.env import Environment
from rl.agent import MLP, ActorCritic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np


# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO', \
    help='Log messages level.')
parser.add_argument('-m', '--model', type=str, default=None, \
    help='Path to trained RL model.')

args = parser.parse_args()


def current_time():
    tz_NY = pytz.timezone('America/New_York') 
    datetime_NY = datetime.now(tz_NY)
    return datetime_NY.strftime("%m_%d_%H:%M:%S")


def calculate_advantages(returns, values, normalize=True):
    advantages = returns - values
    if normalize:        
        advantages = (advantages - advantages.mean()) / advantages.std()        
    return advantages


def update_policy(policy, states, actions, log_prob_actions, \
        advantages, returns, optimizer, ppo_steps, ppo_clip):
    
    total_policy_loss = 0 
    total_value_loss = 0
    
    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()
    
    for _ in range(ppo_steps):
                
        #get new log prob of actions for all input states
        action_pred, value_pred = policy(states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        
        #new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)
        
        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
                
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(
            policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
        
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).sum()
        
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()
    
        optimizer.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        optimizer.step()
    
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


def fl_train(fl_server, policy, optimizer, discount_factor, ppo_steps, ppo_clip):

    # copy client profile
    client_level_accuracy_dict = copy.deepcopy(fl_server.client_level_accuracy_dict)

    rounds = fl_server.config.fl.rounds
    target_accuracy = fl_server.config.fl.target_accuracy

    policy.train()

    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    episode_reward = 0
    done = False
    
    accuracy = 0
    pre_accuracy = 0

    for round_id in range(1, rounds+1):
        logging.info('**** Round {}/{} ****'.format(round_id, rounds))
    
        sample_clients = fl_server.selection()

        state = []
        for client in sample_clients:
            for diff_accuracy in client_level_accuracy_dict[client.client_id]:
                state.append(diff_accuracy)

        # global model accuracy diff
        state.append(round((accuracy - pre_accuracy) * 10000, 2))

        logging.info(f'State: {state}')

        state = torch.FloatTensor(state).reshape(1, -1)
        states.append(state)

        action_pred, value_pred = policy(state)

        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()

        log_prob_action = dist.log_prob(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        actions.append(action)

        pre_accuracy = accuracy # backup current accuracy
        # one round FL training
        accuracy = fl_server.round(
            round_id, action.item(), sample_clients, client_level_accuracy_dict)

        logging.info(f'Clients: {client_level_accuracy_dict}')

        done = target_accuracy and (accuracy >= target_accuracy)

        reward = calculate_reward(accuracy, target_accuracy) # negative value
        logging.info(f'Reward: {reward}\n')
        rewards.append(reward)
        episode_reward += reward        

        if done: # Break loop when target accuracy is met
            logging.info('Target accuracy reached.\n')
            break
    
    states = torch.cat(states)
    actions = torch.IntTensor(actions)
    log_prob_actions = torch.FloatTensor(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    
    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    policy_loss, value_loss = update_policy(
        policy, states, actions, log_prob_actions, \
        advantages, returns, optimizer, ppo_steps, ppo_clip)

    logging.debug(f'Init client dict: {fl_server.client_level_accuracy_dict}')
    logging.debug(f'Runtime client dict: {client_level_accuracy_dict}')

    return policy_loss, value_loss, episode_reward, round_id


def evaluate(fl_server, policy):
    
    policy.eval()

    done = False
    episode_reward = 0

    # copy client profile
    client_level_accuracy_dict = fl_server.client_level_accuracy_dict.copy()
    rounds = fl_server.config.fl.rounds
    target_accuracy = fl_server.config.fl.target_accuracy
    
    accuracy = 0
    pre_accuracy = 0

    for round_id in range(1, rounds+1):
        logging.info('**** Evaluation Round {}/{} ****'.format(round_id, rounds))
    
        sample_clients = fl_server.selection()
        
        state = []
        for client in sample_clients:
            for diff_accuracy in client_level_accuracy_dict[client.client_id]:
                state.append(diff_accuracy)

        # global model accuracy diff
        state.append(round((accuracy - pre_accuracy) * 10000, 2)) 
        logging.info(f'State: {state}')

        state = torch.FloatTensor(state).reshape(1, -1)

        with torch.no_grad():
            action_pred, _ = policy(state)
            action_prob = F.softmax(action_pred, dim = -1)

        action = torch.argmax(action_prob, dim = -1)

        logging.info(f'Choose pruning level {action.item()}')           

        pre_accuracy = accuracy # backup current accuracy
        # one round FL training
        accuracy = fl_server.round(
            round_id, action.item(), sample_clients, client_level_accuracy_dict)

        # logging.info(f'Round {round_id} accuracy: {accuracy}')
        logging.info(f'Clients: {client_level_accuracy_dict}')

        done = target_accuracy and (accuracy >= target_accuracy)

        reward = calculate_reward(accuracy, target_accuracy) # negative value
        logging.info(f'Reward: {reward}\n')
        episode_reward += reward        

        if done: # Break loop when target accuracy is met
            logging.info('Target accuracy reached.\n')
            break
    
    return episode_reward, round_id


def calculate_reward(accuracy, target_accuracy):
    return round((accuracy - target_accuracy) * 1000, 2)


def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0    
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)        
    returns = torch.tensor(returns)    
    if normalize:
        returns = (returns - returns.mean()) / returns.std()        
    return returns


def calculate_advantages(returns, values, normalize=True):
    advantages = returns - values    
    if normalize:        
        advantages = (advantages - advantages.mean()) / advantages.std()        
    return advantages


def update_policy(policy, states, actions, log_prob_actions, \
        advantages, returns, optimizer, ppo_steps, ppo_clip):

    logging.info('Updating policy...\n')

    total_policy_loss = 0 
    total_value_loss = 0
    
    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()
    
    for _ in range(ppo_steps):
                
        #get new log prob of actions for all input states
        action_pred, value_pred = policy(states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        
        #new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)
        
        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
                
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(
            policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
        
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).sum()
        
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()
    
        optimizer.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        optimizer.step()
    
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


def create_fl_server(fl_config):
    # Set logging
    logging.basicConfig(
        filename=os.path.join(
            "/mnt/open_lth_data", 'RL-'+current_time()+'.logger.log'), 
        format='[%(asctime)s][%(levelname)s]: %(message)s', 
        level=logging.INFO, datefmt='%H:%M:%S')
    
    # Clean up global model file saved from last run
    if os.path.exists(os.path.join(fl_config.paths.model, 'global.pth')):
        os.remove(os.path.join(fl_config.paths.model, 'global.pth'))
    
    # Initialize server
    fl_server = server.RLLotteryServer(fl_config) 
    # env = Environment(rl_server)

    fl_server.boot()

    # probe clients
    fl_server.probe()

    logging.info(f'Probing clients: {fl_server.client_level_accuracy_dict}\n')

    return fl_server


def init_rl_model(policy, path_to_model=None):
    logging.info('Initialize RL model')
    if path_to_model:
        logging.info(f'Loading RL model {path_to_model}...')
        policy.load_state_dict(torch.load(path_to_model))
    else:    
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0)
        policy.apply(init_weights)


def main():
    # Read configuration file
    fl_config = config.Config(args.config, args.log)

    fl_server = create_fl_server(fl_config)
    current_run_time = current_time()

    pruning_levels = fl_config.lottery["levels"]
    selected_client_num = fl_config.clients.per_round

    MAX_EPISODES = 1000
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 475
    PRINT_EVERY = 1
    EVAL_EVERY = 5
    PPO_STEPS = 5
    PPO_CLIP = 0.2

    INPUT_DIM = selected_client_num*(pruning_levels+1)+1
    HIDDEN_DIM = 256
    OUTPUT_DIM = pruning_levels+1

    actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

    policy = ActorCritic(actor, critic)

    LEARNING_RATE = 0.0001
    optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

    init_rl_model(policy, args.model)    

    train_rewards = []
    test_rewards = []
    train_round_num = []
    test_round_num = []

    for episode in range(1, MAX_EPISODES+1):
        
        logging.info(f'Episode {episode}')

        fl_server.reset()

        policy_loss, value_loss, train_reward, train_round_id = fl_train(
            fl_server, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP)

        if episode % EVAL_EVERY == 0:
            fl_server.reset()        
            test_reward, test_round_id = evaluate(fl_server, policy)
            
            test_rewards.append(test_reward)
            test_round_num.append(test_round_id)
            mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
            mean_test_round_num = np.mean(test_round_num)

        train_rewards.append(train_reward)
        train_round_num.append(train_round_id)
        
        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_train_round_num = np.mean(train_round_num)
        
        if episode % PRINT_EVERY == 0:
            logging.info(
                f'| Episode: {episode:3} | '\
                +f'Train Rewards: {train_reward:5.1f} | '\
                +f'Train Round Num: {train_round_id} |\n')
        
        if episode % EVAL_EVERY == 0:
            logging.info(
                f'| Episode: {episode:3} | '\
                +f'Test Rewards: {test_reward:5.1f} | '\
                +f'Test Round Num: {test_round_id} |\n')
        
            if mean_test_rewards >= REWARD_THRESHOLD:            
                logging.info(f'Reached reward threshold in {episode} episodes')
                break
        
        torch.save(policy.state_dict(), os.path.join(fl_config.paths.model, \
                f'rl-{fl_config.lottery["model_name"]}-{current_run_time}.pth'))
            
 
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
