# calib_config.py

import configparser
import os


class CalibConfig:
    """
    Wrapper around ConfigParser that:
      - checks file existence,
      - parses values into bool/int/float/list when possible,
      - exposes a simple .get(section, key, default) API.
    """

    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

        parser = configparser.ConfigParser()
        parser.optionxform = str  # preserve case sensitivity of keys
        read_files = parser.read(config_path)

        if not read_files:
            raise IOError(f"Failed to read configuration file '{config_path}'.")

        self._parser = parser
        self.config = self._parse_config()

    def _parse_config(self) -> dict:
        cfg: dict[str, dict[str, object]] = {}

        for section in self._parser.sections():
            cfg[section] = {}
            for key, val in self._parser.items(section):
                val_stripped = val.strip()

                # Bool
                if val_stripped.lower() in {"true", "false"}:
                    parsed_val = val_stripped.lower() == "true"
                # List (comma-separated)
                elif "," in val_stripped:
                    parsed_val = [v.strip() for v in val_stripped.split(",")]
                else:
                    # Try int
                    try:
                        parsed_val = int(val_stripped)
                    except ValueError:
                        # Try float
                        try:
                            parsed_val = float(val_stripped)
                        except ValueError:
                            parsed_val = val_stripped
                cfg[section][key] = parsed_val

        return cfg

    def get(self, section: str, key: str, default=None):
        return self.config.get(section, {}).get(key, default)

    def get_section(self, section: str) -> dict:
        return self.config.get(section, {})
