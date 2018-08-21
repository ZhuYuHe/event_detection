import argparse
from model.config import Config

def add_argument(parser):
    """Build ArgumentParser."""
    # input and output
    parser.add_argument('--model', type=str, default='textcnn', choices=['textcnn', 'attention'], help="model used to train")
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='checkpoint directory path')

def config_from_args(args):
    """build config and read from args"""
    config = Config()

    return config
