'''
@ref: Incorporating Query Reformulating Behavior into Web Search Evaluation
@author: Jia Chen, Yiqun Liu, Jiaxin Mao, Fan Zhang, Tetsuya Sakai, Weizhi Ma, Min Zhang, Shaoping Ma
@desc: Configurations and startups
'''

# encoding:utf-8
import argparse
import logging
from model import *


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('RAM')

    parser.add_argument('--click_model', default='DBN', type=str, help='choose the click model for derivation, options=["DBN", "UBM", "SDBN", "PBM"]')

    parser.add_argument('--data', default='qref', type=str, help='choose the data set, options=["qref","fsd"]')

    parser.add_argument('--id', default=0, type=int, help='bootstrap id, [0 - 99]')

    parser.add_argument('--metric_type', default='expected_utility', type=str, help='choose the type for click model-based metric, options=["expected_utility", "effort"]')

    parser.add_argument('--max_usefulness', default=3, type=int, help='the maximum usefulness label of the data set')

    parser.add_argument('--k_num', default=6, type=int, help='the number of reformulation types to consider')

    parser.add_argument('--max_dnum', default=10, type=int, help='the maximum number of documents under a query')

    # training settings
    train_settings = parser.add_argument_group('train settings')

    train_settings.add_argument('--iter_num', default=1e4, type=int, help='number of training iterations')

    train_settings.add_argument('--use_knowledge', default=False, type=bool, help='whether use the transition probability matrix from TianGong-Qref')

    train_settings.add_argument('--alpha', default=1e-2, type=float, help='initial learning rate')

    train_settings.add_argument('--lamda', default=1, type=float, help='the weight for satisfaction prediction')

    train_settings.add_argument('--alpha_decay', default=0.99, type=float, help='learning rate decay')

    train_settings.add_argument('--patience', default=5, type=int, help='lr half when more than the patience times of evaluation')

    return parser.parse_args()


def train(args):
    """
    trains the RAM
    """
    logger = logging.getLogger("RAM")
    logger.info('Initialize the model...')

    model_name = args.click_model
    if model_name in ['DBN', 'SDBN']:
        model = uDBN(args)
    elif model_name == 'UBM':
        model = uUBM(args)
    elif model_name == 'PBM':
        model = uPBM(args)
    logger.info('Training the model...')
    model.train_model()
    logger.info('Done with model training!')
    model.eval()


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()
    assert args.click_model in ['UBM', 'PBM', 'DBN', 'SDBN']
    assert args.metric_type in ['expected_utility', 'effort']
    if args.click_model == 'UBM' or args.click_model == 'PBM':
        assert args.metric_type == 'expected_utility'

    # create a logger
    logger = logging.getLogger("RAM")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))
    train(args)

    logger.info('run done.')


if __name__ == '__main__':
    run()
