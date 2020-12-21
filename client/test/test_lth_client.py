import sys

sys.path.append("..")
sys.path.append("../open_lth/")
sys.path.append("../../")

from lth_client import LTHClient
import config

lth_config = config.Config(
    "/home/ubuntu/flsim-lottery/configs/Lottery/mnist.json")

lth_clt = LTHClient(0)
lth_clt.configure(lth_config)
lth_clt.train()
