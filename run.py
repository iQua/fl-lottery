import argparse
import client
import config
import os
import server
import sys

import torch

# sys.path.append("./client/open_lth")

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()


def main():
    """Run a federated learning simulation."""

    # Read configuration file
    fl_config = config.Config(args.config, args.log)

    # Initialize server
    fl_server = {
        "basic": server.Server(fl_config),
        "accavg": server.AccAvgServer(fl_config),
        "directed": server.DirectedServer(fl_config),
        "kcenter": server.KCenterServer(fl_config),
        "kmeans": server.KMeansServer(fl_config),
        "magavg": server.MagAvgServer(fl_config),
        "lth": server.LotteryServer(fl_config) 
    }[fl_config.server]
    
    fl_server.boot()

    # Run federated learning
    fl_server.run()

    # Delete global model
    os.remove(os.path.join(fl_config.paths.model, 'global.pth'))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
