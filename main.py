import argparse
import argh
from time import time
from contextlib import contextmanager
import os
import random
import re
import sys
from collections import namedtuple

import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

from config import FLAGS, HPS
from dataset import get_batch


@contextmanager
def timer(message: str):
    tick = time()
    yield
    tock = time()
    logger.info(f'{message}: {(tock - tick):.3f} seconds')


def train(flags=FLAGS, hps=HPS):
    from Network import Net
    net = Net(flags, hps)

    for g in range(flags.global_epoch):
        with timer(f'Global epoch #{g}'):
            logger.debug(f'Start global epoch {g}')
            net.train(porportion=0.01)
            l, acc = net.test(porportion=0.1)
            logger.debug(f'Finish global epoch {g}')

    net.save_model(name=f'{l:.4f}-{acc:.4f}')
    logger.info('All done')


def test(flags=FLAGS, hps=HPS):
    pass


if __name__ == "__main__":

    if not os.path.exists('./train_log'):
        os.makedirs('./train_log')

    if not os.path.exists('./test_log'):
        os.makedirs('./test_log')

    if not os.path.exists('./savedmodels'):
        os.makedirs('./savedmodels')

    fn = {'train': lambda: train(),
          'test': lambda: test()}

    if fn.get(FLAGS.MODE, 0) != 0:
        fn[FLAGS.MODE]()
    else:
        logger.info('Please choose a mode among "train", "test".')
