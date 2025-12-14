import yaml
import logging

logger = logging.getLogger("set_config")


def load_yaml_conf(config_path):

    with open(config_path, "r", encoding="utf-8") as file:
        args = yaml.safe_load(file)
    return args


def write_yaml_conf(config_path, data):
    # Save the updated configuration to a new YAML file
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, default_flow_style=False)
