import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, TypeVar

import omegaconf
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from pydantic import BaseModel

from tiny_moves.utils.paths import get_configs_dir

from .hydra_plugins import register_plugins

# create type T that is a subclass of BaseModel
T = TypeVar("T", bound=BaseModel)
logging.basicConfig(level=logging.ERROR)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """
    Parse command-line arguments and separate known arguments from Hydra overrides.

    Args:
        description: Description of the command-line arguments.

    Returns:
        A tuple (args, overrides) where:
          - args: Parsed arguments for config_path and config_name.
          - overrides: Additional command-line parameters for Hydra.

    """
    parser = argparse.ArgumentParser(
        description="A command line interface for games played by multi-agent systems."
    )
    parser.add_argument("--experiment_name", default="Default", help="Name of experiment to be tracked in MLFlow.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(get_configs_dir("entry_points")),
        help="Path to the configuration directory",
    )
    parser.add_argument(
        "--config_name", type=str, default="chat_config", help="Name of the top-level config file (without extension)"
    )

    # Parse known arguments; any remaining ones are treated as Hydra overrides.
    args, overrides = parser.parse_known_args()
    return args, overrides


def load_config(
    config_path: str,
    config_name: str,
    overrides: list[str],
    hydra_config_cls: type[T],
) -> T:
    """
    Load and compose a Hydra configuration and convert it into a Pydantic model.

    This function does the following:
      1. Normalizes the provided config_path so that it is interpreted relative to the project root.
      2. Initializes Hydra using that normalized path.
      3. Composes the configuration from the specified top-level config file (config_name)
         while applying any command-line overrides.
      4. Converts the resulting OmegaConf object into an instance of hydra_config_cls using Pydantic.

    Args:
        config_path: The configuration directory path, intended to be relative to the project root.
        config_name: The name of the top-level config file (without the file extension).
        overrides: A list of command-line overrides to pass to Hydra.
        hydra_config_cls: The Pydantic model class to validate and instantiate the configuration.

    Returns:
        An instance of hydra_config_cls containing the fully composed configuration.

    """
    # Normalize the config_path so Hydra finds the directory correctly.
    normalized_path = normalize_config_path(config_path, __file__)
    register_plugins()
    with initialize(config_path=normalized_path, version_base="1.1"):
        cfg = compose(config_name=config_name, overrides=overrides)

        try:
            cfg = instantiate(cfg)
        except omegaconf.errors.MissingMandatoryValue as e:
            print_missing_params(cfg)
            raise e

    # Convert the OmegaConf configuration to a dict and validate it using the Pydantic model.
    chat_config = hydra_config_cls.model_validate(OmegaConf.to_container(cfg, resolve=True))
    return chat_config


def print_missing_params(cfg: DictConfig) -> None:
    """
    Walk through the OmegaConf config and print out any keys that are still missing.

    Args:
        cfg: The OmegaConf configuration object.

    """
    missing = {}

    def recurse(conf: Any, prefix: str = "") -> None:
        if isinstance(conf, dict):
            for k, v in conf.items():
                key_path = f"{prefix}.{k}" if prefix else k
                if v == MISSING:
                    missing[key_path] = "MISSING (required)"
                elif isinstance(v, dict) or OmegaConf.is_config(v):
                    recurse(v, key_path)
        elif isinstance(conf, list):
            for idx, item in enumerate(conf):
                key_path = f"{prefix}[{idx}]"
                if item == MISSING:
                    missing[key_path] = "MISSING (required)"
                elif isinstance(item, dict) or OmegaConf.is_config(item):
                    recurse(item, key_path)

    recurse(OmegaConf.to_container(cfg, resolve=False))

    if missing:
        print("The following required parameters are missing:")
        for key, msg in missing.items():
            print(f"  {key}: {msg}")
    else:
        print("All required parameters are set.")


def normalize_config_path(config_path: str, file_path: str) -> str:
    """
    Normalize a configuration directory path for Hydra.

    Hydra.initialize() resolves the given config_path relative to the directory of the caller
    (i.e. the file where initialize is invoked). In many projects the desired config_path is meant to be
    relative to the project root rather than this caller's location.

    This function converts a config_path intended to be relative to the project root into a path that is
    relative to the caller's file location.

    For example, consider this project structure:

        <project_root>/
            configs/              <-- configuration directory (project-root-relative)
            tiny_moves/
                utils/
                    entry_points/
                        cli_hydra_setup.py  <-- This file

    If config_path is "configs" (i.e. referring to <project_root>/configs) and this file is located at:
        <project_root>/tiny_moves/utils/entry_points/cli_hydra_setup.py
    then this function calculates the relative path from:
        <project_root>/tiny_moves/utils/entry_points
    to:
        <project_root>/configs
    which would be "../../configs".

    Note: Adjust the index in `parents[...]` if your project structure differs.

    Args:
        config_path: The configuration directory as intended relative to the project root.
        file_path: The __file__ value of the caller file.

    Returns:
        A relative path from the caller's directory to the desired configuration directory.

    """
    # Determine the project root.
    # Here we assume that the project root is 3 levels up from the caller file.
    project_root = Path(file_path).parents[3]

    # Compute the absolute path of the desired configuration directory.
    desired_config_path = (project_root / config_path).resolve()

    # Determine the directory of the caller.
    caller_dir = Path(file_path).parent

    # Compute the relative path from the caller's directory to the desired config directory.
    normalized_path = os.path.relpath(desired_config_path, caller_dir)
    return normalized_path


ChatConfigType = TypeVar("ChatConfigType", bound=BaseModel)


def cli(config_cls: type[ChatConfigType], args: argparse.Namespace, overrides: list[str]) -> ChatConfigType:
    """Run the CLI argument parsing and chat based on the provided configuration."""
    try:
        print(args)
        chat_config = load_config(args.config_path, args.config_name, overrides, config_cls)
    except Exception as e:
        logging.error(f"Unexpected error while parsing config: {e}", exc_info=True)
        sys.exit()

    return chat_config
