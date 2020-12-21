# Federated Learning with Lottery Tickets



### FedLottery: Optimizing Federated Learning with Lottery Tickets

Federated Learning has become a practical distributed machine learning paradigm that exploits massive volumes of user data stored on mobile devices to train a global model without violating user privacy. However, unreliable and relatively slow network connections from mobile devices to the server have become a critical bottleneck for model updates, potentially degrading its performance. Without access to training data, it is infeasible to optimize the global model structure and shrink model updates without degrading model quality.
  
In this work, we propose *FedLottery*, a new communication-efficient federated learning framework. *FedLottery* accelerates federated learning by jointly reducing the total number of communication rounds and the per-round communication overhead. By adapting the lottery ticket hypothesis in federated learning, we can identify a key model substructure---the winning ticket---that improves the training efficiency with fewer model parameters. We further propose a new  aggregation algorithm based on deep reinforcement learning (DRL). Our algorithm is designed to choose the appropriate winning tickets and to update the global model, aiming for an optimal balance between the model quality and size. With cautiously crafted observation states and rewards, we train our DRL agent with the actor-critic proximal policy optimization (Actor-Critic PPO) algorithm that steadily learns from the dynamic federated learning context. 

