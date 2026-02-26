import pathlib as pl

from bagel.config.compiler import validate_loaded_config
from bagel.config.loader import load_config


def test_agreed_baseline_config_is_valid_static() -> None:
    config_path = pl.Path(__file__).resolve().parents[1] / 'fixtures' / 'configs' / 'agreed_baseline.yaml'
    loaded = load_config(config_path)
    validate_loaded_config(loaded)
