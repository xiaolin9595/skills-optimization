import logging
import sys
from typing import Optional

def setup_logger(name: str = "skill_opt", level: str = "INFO") -> logging.Logger:
    """
    Sets up a logger with standard formatting.
    
    Args:
        name: The name of the logger.
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        
    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
        
    logger.setLevel(numeric_level)
    
    # Check if handler already exists to avoid duplicate logs
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# Default global logger
logger = setup_logger()
