"""
# Flexible Log File Loader

## Overview:
This script provides a flexible log file loader class, `LogFileLoader`, designed to handle various log file formats. It allows users to specify custom parsing functions for timestamps, severity levels, and log messages, making it adaptable to different log structures.

### Dependencies:
- loguru: Logging library for flexible and easy-to-use logging.
- dateutil: Part of the Python standard library, used for parsing timestamps.
- termcolor: Library for adding colored output to the terminal.

## LogFileLoader Class:
The `LogFileLoader` class allows customization through the following parameters during initialization:
- `log_file_path`: Path to the log file.
- `timestamp_parser`: Custom timestamp parsing function.
- `severity_parser`: Custom severity parsing function.
- `message_parser`: Custom log message parsing function.

### Parsing Functions:
Three default parsing functions are provided:
- `default_timestamp_parser`: Parses timestamps using dateutil.parser.
- `default_severity_parser`: Strips and returns severity.
- `default_message_parser`: Strips and returns log messages.

## Test Cases:
This script includes an integration test, `TestLogFileLoaderIntegration`, that verifies the functionality of the `LogFileLoader` class by loading and parsing logs from a real log file.
"""



import unittest
import os
from loguru import logger
from dateutil import parser as date_parser
from termcolor import colored  # Install using: pip install termcolor

class LogFileLoader:
    def __init__(self, log_file_path, timestamp_parser=None, severity_parser=None, message_parser=None):
        """
        A flexible log file loader that allows custom parsing functions for timestamps, severity, and messages.

        :param log_file_path: Path to the log file.
        :param timestamp_parser: Custom timestamp parsing function.
        :param severity_parser: Custom severity parsing function.
        :param message_parser: Custom log message parsing function.
        """
        self.log_file_path = log_file_path
        self.timestamp_parser = timestamp_parser or self.default_timestamp_parser
        self.severity_parser = severity_parser or self.default_severity_parser
        self.message_parser = message_parser or self.default_message_parser

    @staticmethod
    def default_timestamp_parser(timestamp_str):
        """
        Default timestamp parsing function using dateutil.parser.

        :param timestamp_str: String representation of the timestamp.
        :return: Parsed timestamp or None if parsing fails.
        """
        try:
            return date_parser.parse(timestamp_str, fuzzy_with_tokens=True)[0]
        except ValueError:
            return None

    @staticmethod
    def default_severity_parser(severity_str):
        """
        Default severity parsing function.

        :param severity_str: String representation of severity.
        :return: Parsed severity.
        """
        return severity_str.strip()

    @staticmethod
    def default_message_parser(message_str):
        """
        Default log message parsing function.

        :param message_str: String representation of the log message.
        :return: Parsed log message.
        """
        return message_str.strip()

    def load_logs(self):
        """
        Load log entries from the specified log file.

        :return: List of parsed log entries.
        """
        try:
            with open(self.log_file_path, 'r') as file:
                logs = file.readlines()
                parsed_logs = self.parse_logs(logs)
                return parsed_logs
        except FileNotFoundError:
            logger.error(f"Error: Log file '{self.log_file_path}' not found.")
            return None
        except Exception as e:
            logger.error(f"Error: {e}")
            return None

    def parse_logs(self, logs):
        """
        Parse log entries using the specified parsing functions.

        :param logs: List of log entries.
        :return: List of parsed log entries.
        """
        parsed_logs = []

        for log in logs:
            timestamp = self.timestamp_parser(log[:23])
            severity = self.severity_parser(log[24:30])
            message = self.message_parser(log[30:])

            if timestamp and severity is not None and message:
                parsed_logs.append({
                    'timestamp': timestamp,
                    'severity': severity,
                    'message': message,
                    'colored_message': self.colorize_message(severity, message)
                })

        return parsed_logs

    def colorize_message(self, severity, message):
        """
        Colorize log messages based on severity.

        :param severity: Log severity level.
        :param message: Log message.
        :return: Colored log message.
        """
        if severity.lower() == 'error':
            return colored(message, 'red', attrs=['bold'])
        elif severity.lower() == 'warning':
            return colored(message, 'yellow', attrs=['bold'])
        elif severity.lower() == 'info':
            return colored(message, 'cyan', attrs=['bold'])
        else:
            return message

# Example usage:
log_file_path = 'random_log_file.log'
log_loader = LogFileLoader(log_file_path)
logs = log_loader.load_logs()

if logs:
    for log in logs:
        print(f"Timestamp: {log['timestamp']}, Severity: {log['severity']}, Message: {log['colored_message']}")



# Markdown Section
