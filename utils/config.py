"""
Configuration and API key management for the Multi-Modal AI system.

This module provides configuration loading, secure API key management, and environment
variable handling for finance and climate domains.
"""
import os
import json
import logging
import configparser
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration paths
DEFAULT_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "scientific_viz_ai")
DEFAULT_CONFIG_FILE = os.path.join(DEFAULT_CONFIG_DIR, "config.ini")
DEFAULT_API_FILE = os.path.join(DEFAULT_CONFIG_DIR, "api_keys.json")

class ConfigManager:
    """
    Configuration manager for the Scientific Visualization AI system.
    """
    def __init__(
        self,
        config_file: Optional[str] = None,
        api_file: Optional[str] = None,
        create_if_missing: bool = True
    ):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the configuration file
            api_file: Path to the API keys file
            create_if_missing: Whether to create missing configuration files
        """
        self.config_file = config_file or DEFAULT_CONFIG_FILE
        self.api_file = api_file or DEFAULT_API_FILE
        
        # Create config directory if it doesn't exist
        config_dir = os.path.dirname(self.config_file)
        if not os.path.exists(config_dir) and create_if_missing:
            os.makedirs(config_dir, exist_ok=True)
            logger.info(f"Created configuration directory: {config_dir}")
            
        # Load or create configuration
        self.config = configparser.ConfigParser()
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
            logger.debug(f"Loaded configuration from {self.config_file}")
        elif create_if_missing:
            self._create_default_config()
            
        # Load or create API keys
        self.api_keys = {}
        if os.path.exists(self.api_file):
            try:
                with open(self.api_file, 'r') as f:
                    self.api_keys = json.load(f)
                logger.debug(f"Loaded API keys from {self.api_file}")
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in API keys file {self.api_file}")
                if create_if_missing:
                    self._create_default_api_keys()
        elif create_if_missing:
            self._create_default_api_keys()
    
    def _create_default_config(self) -> None:
        """Create a default configuration file."""
        self.config["DEFAULT"] = {
            "data_dir": os.path.join(os.path.expanduser("~"), "scientific_viz_ai_data"),
            "log_level": "INFO",
            "use_gpu": "True",
            "cache_dir": os.path.join(os.path.expanduser("~"), ".cache", "scientific_viz_ai")
        }
        
        self.config["visualization"] = {
            "default_format": "png",
            "dpi": "300",
            "interactive": "True"
        }
        
        self.config["finance"] = {
            "api_source": "yahoo_finance",
            "default_ticker": "AAPL",
            "default_period": "1y",
            "default_indicators": "SMA,EMA,RSI"
        }
        
        self.config["climate"] = {
            "api_source": "noaa",
            "default_dataset": "GHCND",
            "default_region": "global",
            "spatial_resolution": "1.0"
        }
        
        self.save_config()
        logger.info(f"Created default configuration file: {self.config_file}")
    
    def _create_default_api_keys(self) -> None:
        """Create a default API keys file."""
        self.api_keys = {
            "yahoo_finance": {
                "api_key": os.environ.get("YAHOO_FINANCE_API_KEY", "")
            },
            "alpha_vantage": {
                "api_key": os.environ.get("ALPHA_VANTAGE_API_KEY", "")
            },
            "noaa": {
                "api_key": os.environ.get("NOAA_API_KEY", "")
            },
            "openweathermap": {
                "api_key": os.environ.get("OPENWEATHERMAP_API_KEY", "")
            }
        }
        
        self.save_api_keys()
        logger.info(f"Created default API keys file: {self.api_file}")
    
    def save_config(self) -> None:
        """Save the current configuration to file."""
        with open(self.config_file, 'w') as f:
            self.config.write(f)
        logger.debug(f"Saved configuration to {self.config_file}")
    
    def save_api_keys(self) -> None:
        """Save the current API keys to file."""
        api_dir = os.path.dirname(self.api_file)
        if not os.path.exists(api_dir):
            os.makedirs(api_dir, exist_ok=True)
            
        with open(self.api_file, 'w') as f:
            json.dump(self.api_keys, f, indent=2)
        
        # Set restrictive permissions for the API keys file
        os.chmod(self.api_file, 0o600)  # Read/write only for the owner
        logger.debug(f"Saved API keys to {self.api_file} with restricted permissions")
    
    def get_config(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if the key doesn't exist
            
        Returns:
            Configuration value
        """
        try:
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def set_config(self, section: str, key: str, value: str) -> None:
        """
        Set a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Configuration value
        """
        if not self.config.has_section(section):
            self.config.add_section(section)
        
        self.config.set(section, key, str(value))
        logger.debug(f"Set config {section}.{key} = {value}")
    
    def get_api_key(self, service: str, env_var: Optional[str] = None) -> Optional[str]:
        """
        Get an API key for a service.
        
        Args:
            service: Service name
            env_var: Environment variable to check first
            
        Returns:
            API key if available, None otherwise
        """
        # First check environment variable if provided
        if env_var and env_var in os.environ:
            return os.environ[env_var]
        
        # Then check standard environment variable pattern
        std_env_var = f"{service.upper()}_API_KEY"
        if std_env_var in os.environ:
            return os.environ[std_env_var]
        
        # Finally check API keys file
        if service in self.api_keys:
            return self.api_keys[service].get("api_key")
        
        return None
    
    def set_api_key(self, service: str, api_key: str, save: bool = True) -> None:
        """
        Set an API key for a service.
        
        Args:
            service: Service name
            api_key: API key
            save: Whether to save the API keys to file
        """
        if service not in self.api_keys:
            self.api_keys[service] = {}
        
        self.api_keys[service]["api_key"] = api_key
        logger.debug(f"Set API key for {service}")
        
        if save:
            self.save_api_keys()
    
    def get_yahoo_finance_api_key(self) -> Optional[str]:
        """
        Get the Yahoo Finance API key.
        
        Returns:
            API key if available, None otherwise
        """
        return self.get_api_key("yahoo_finance", "YAHOO_FINANCE_API_KEY")
    
    def set_yahoo_finance_api_key(self, api_key: str, save: bool = True) -> None:
        """
        Set the Yahoo Finance API key.
        
        Args:
            api_key: API key
            save: Whether to save the API keys to file
        """
        self.set_api_key("yahoo_finance", api_key, save)
        
    def get_noaa_api_key(self) -> Optional[str]:
        """
        Get the NOAA API key.
        
        Returns:
            API key if available, None otherwise
        """
        return self.get_api_key("noaa", "NOAA_API_KEY")
    
    def set_noaa_api_key(self, api_key: str, save: bool = True) -> None:
        """
        Set the NOAA API key.
        
        Args:
            api_key: API key
            save: Whether to save the API keys to file
        """
        self.set_api_key("noaa", api_key, save)
    
    def create_data_directories(self) -> Dict[str, str]:
        """
        Create necessary data directories based on configuration.
        
        Returns:
            Dictionary mapping directory types to paths
        """
        dirs = {}
        
        # Base data directory
        data_dir = self.get_config("DEFAULT", "data_dir", 
                                  os.path.join(os.path.expanduser("~"), "scientific_viz_ai_data"))
        dirs["data"] = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Cache directory
        cache_dir = self.get_config("DEFAULT", "cache_dir",
                                   os.path.join(os.path.expanduser("~"), ".cache", "scientific_viz_ai"))
        dirs["cache"] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Subdirectories
        for subdir in ["finance", "climate", "visualizations", "models"]:
            path = os.path.join(data_dir, subdir)
            dirs[subdir] = path
            os.makedirs(path, exist_ok=True)
        
        logger.info(f"Created data directories: {', '.join(dirs.keys())}")
        return dirs


# Global configuration instance
config_manager = ConfigManager()


def get_api_key(service: str) -> Optional[str]:
    """
    Get an API key for a service.
    
    Args:
        service: Service name
        
    Returns:
        API key if available, None otherwise
    """
    return config_manager.get_api_key(service)


def setup_api_key_prompt() -> None:
    """Interactive prompt to set up API keys."""
    print("\n=== API Key Setup ===")
    
    # Yahoo Finance API key
    yf_key = config_manager.get_yahoo_finance_api_key()
    if not yf_key:
        print("\nA Yahoo Finance API key may be required for downloading financial data.")
        print("You can get an API key from RapidAPI Yahoo Finance API")
        yf_key = input("Enter your Yahoo Finance API key (press Enter to skip): ")
        if yf_key:
            config_manager.set_yahoo_finance_api_key(yf_key)
            print("Yahoo Finance API key saved!")
        else:
            print("Skipped Yahoo Finance API key setup.")
    
    # NOAA API key
    noaa_key = config_manager.get_noaa_api_key()
    if not noaa_key:
        print("\nA NOAA API key may be required for downloading climate data.")
        print("You can get an API key at https://www.ncdc.noaa.gov/cdo-web/token")
        noaa_key = input("Enter your NOAA API key (press Enter to skip): ")
        if noaa_key:
            config_manager.set_noaa_api_key(noaa_key)
            print("NOAA API key saved!")
        else:
            print("Skipped NOAA API key setup.")
    
    print("\nAPI key setup complete. Keys are stored in:", config_manager.api_file)
    print("You can also set API keys using environment variables:")
    print("  - Yahoo Finance: YAHOO_FINANCE_API_KEY")
    print("  - Alpha Vantage: ALPHA_VANTAGE_API_KEY")
    print("  - NOAA: NOAA_API_KEY")
    print("  - OpenWeatherMap: OPENWEATHERMAP_API_KEY")
    print("")