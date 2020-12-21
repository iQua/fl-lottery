



class Environment(object):
    def __init__(self, fl_server):
        
        self.fl_server = fl_server
        self.fl_server.boot()

        # Run federated learning
        self.fl_server.run()


    def reset(self):
        pass
        
        # Delete global model
        os.remove(self.fl_server.paths.model + '/global')


    def step(self, action):
        pass


    def get_reward(self):
        pass


    def plot_state(self):
        pass


    def observe(self):
        pass