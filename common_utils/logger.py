"""
Logging utilities for MMDG-Bench

Provides Logger class that writes output to both console and file.
"""

import sys
from datetime import datetime
import os


class Logger:
    """
    Logger that writes to both console and file

    Usage:
        logger = Logger('training.log')
        sys.stdout = logger
        sys.stderr = logger

        print("This goes to both console and file")

        # Don't forget to close
        logger.close()
    """
    def __init__(self, log_file):
        """
        Args:
            log_file: Path to log file
        """
        self.terminal = sys.stdout

        # Create directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.log = open(log_file, 'w', buffering=1)  # Line buffered

        # Write header
        self.log.write("="*80 + "\n")
        self.log.write(f"Log started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write("="*80 + "\n\n")
        self.log.flush()

    def write(self, message):
        """Write message to both terminal and file"""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write

    def flush(self):
        """Flush both terminal and file"""
        self.terminal.flush()
        self.log.flush()

    def close(self):
        """Close log file and restore stdout"""
        if self.log:
            self.log.write("\n" + "="*80 + "\n")
            self.log.write(f"Log ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log.write("="*80 + "\n")
            self.log.close()
            self.log = None


def setup_logging(log_file):
    """
    Convenience function to setup logging to both console and file

    Args:
        log_file: Path to log file

    Returns:
        logger: Logger instance

    Usage:
        logger = setup_logging('training.log')
        print("This goes to both console and file")
        # ... do your work ...
        logger.close()
    """
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    return logger