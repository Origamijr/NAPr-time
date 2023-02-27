from utils import is_interactive
import argparse
import toml

_config_path = "./config.toml"
if not is_interactive():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--config', default='./config.toml', type=str)
    args = _parser.parse_args()
    _config_path = args.config

def reload():
    # To use in an interractive environment to relead config object when file is changed.
    global _CONFIG
    _CONFIG = toml.load(_config_path)

def config():
    return _CONFIG

def data_config():
    return _CONFIG['data']

def feature_config():
    return data_config()['features']

def training_config():
    return _CONFIG['training']

reload()