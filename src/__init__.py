"""Dummy Docstring."""
import logging
import sys

# Set up logging
logging.basicConfig(
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    force=True,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
