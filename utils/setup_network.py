from models.vgg import Vgg
from models.vgg_baseline import VggBaseline
from utils.checkpoint import restore_checkpoint
from utils.logger import Logger

nets = {
    'vgg': Vgg,
    'vgg_baseline': VggBaseline,
}


def setup_network(hps):
    if hps['network'] == 'vgg_baseline':
        net = nets[hps['network']]()
    else:
        net = nets[hps['network']](drop=hps['drop'])

    logger = Logger()
    if hps['restore_epoch']:
        restore_checkpoint(net, logger, hps)

    return logger, net
