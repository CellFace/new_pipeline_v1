# coding=utf-8

# Standard Library Imports
import os
import json
import logging
from logging.handlers import RotatingFileHandler


def create_dir(directory='./output'):
    """
        Creates a directory if it does not already exist.

        Args:
            directory (str, optional): The path of the directory to create. 
                                    Default is './output'.

        Returns:
            str: The path of the created or existing directory.

        Process:
            - Checks if the specified directory exists.
            - If the directory does not exist, it creates it using `os.makedirs()`.
            - Returns the directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


def setup_logger(name, log_dir='./log', log_filename='dhm.log', level=logging.INFO, max_size=10485760, backup_count=5):
    """
        Sets up and returns a logger with a rotating file handler.

        Args:
            name (str): The name of the logger.
            log_dir (str, optional): The directory where log files will be stored. Default is './log'.
            log_filename (str, optional): The name of the log file. Default is 'dhm.log'.
            level (int, optional): The logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.
            max_size (int, optional): The maximum file size (in bytes) before log rotation occurs. Default is 10MB.
            backup_count (int, optional): The number of backup log files to keep. Default is 5.

        Returns:
            logging.Logger: A configured logger instance.

        Process:
            - Ensures the log directory exists by calling `create_dir(log_dir)`.
            - Defines the log file path using `log_dir` and `log_filename`.
            - Creates a log formatter with timestamp, log level, and message format.
            - Initializes a `RotatingFileHandler` to handle log rotation based on file size.
            - Retrieves or creates a logger with the given name and sets its logging level.
            - Clears existing handlers (if any) to prevent duplicate log entries.
            - Adds the rotating file handler to the logger.
            - Returns the configured logger instance.
    """
    # Create the log directory if it doesn't exist
    create_dir(directory=log_dir)

    log_filepath = os.path.join(log_dir, log_filename)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler = RotatingFileHandler(log_filepath, maxBytes=max_size, backupCount=backup_count)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if the logger already has handlers and remove them if necessary
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

    return logger


def load_config(config_file="./config/config.json"):
    """
        Loads a configuration file in JSON format.

        Args:
            config_file (str, optional): The path to the configuration JSON file.
                                        Default is "./config/config.json".

        Returns:
            dict: A dictionary containing the configuration data.

        Process:
            - Determines the absolute path of the script's base directory.
            - If `config_file` is None, sets the default path to `../config/config.json`.
            - Constructs the absolute path to the configuration file.
            - Opens and reads the JSON file.
            - Parses and returns the JSON content as a dictionary.
    """
    # Get the absolute path to the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if config_file is None:
        config_file = os.path.join(base_dir, '..', 'config', 'config.json')
        
    with open(os.path.abspath(os.path.join(base_dir, '..', config_file))) as f:
        config = json.load(f)
    
    return config
