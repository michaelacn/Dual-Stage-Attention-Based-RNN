import os 
from easydict import EasyDict as edict
import yaml
import json
import errno


class Loader(yaml.SafeLoader):
    """
    YAML Loader with `!include` constructor.
    """
    def __init__(self, stream) -> None:
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader, node: yaml.Node):
    """
    Include file referenced at node.
    """
    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())


def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config


def create_checkpoint_directory(checkpoint_path):
    """
    Create a checkpoint directory if it doesn't exist.
    """
    try:
        os.mkdir(checkpoint_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', checkpoint_path)
