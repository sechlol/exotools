import contextlib
import logging
import warnings

from astropy.units import UnitParserWarning, UnitsWarning


@contextlib.contextmanager
def units_warnings_as_exceptions():
    """
    Context manager that temporarily converts astropy unit-related warnings into exceptions.

    This allows for tracing the origin of warnings like:
    "WARNING: UnitsWarning: 'days' did not parse as unit: At col 0, days is not a valid unit."

    Usage:
        with units_warnings_as_exceptions():
            # Code that might generate unit warnings
            # Any UnitsWarning or UnitParserWarning will be raised as exceptions

    Raises:
        Exception: If any UnitsWarning or UnitParserWarning is triggered within the context
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=UnitsWarning)
        warnings.filterwarnings("error", category=UnitParserWarning)
        yield


@contextlib.contextmanager
def silence_warnings(warning_types=None):
    """
    Context manager that temporarily silences specific warnings or all warnings.

    Args:
        warning_types: List of warning types to silence. If None, all warnings are silenced.

    Usage:
        # Silence specific warning types
        with silence_warnings([UnitsWarning, UnitParserWarning]):
            # Code that might generate warnings
            # The specified warnings will be silenced

        # Silence all warnings
        with silence_warnings():
            # Code that might generate warnings
            # All warnings will be silenced
    """
    with warnings.catch_warnings():
        if warning_types:
            for warning_type in warning_types:
                warnings.filterwarnings("ignore", category=warning_type)
        else:
            warnings.filterwarnings("ignore")
        yield


class LoggingExceptionHandler(logging.Handler):
    """
    A logging handler that raises exceptions for log messages at or above a specified level.
    """

    def __init__(self, level=logging.WARNING):
        super().__init__(level)

    def emit(self, record):
        message = self.format(record)
        raise Exception(f"Log message converted to exception: {message}")


@contextlib.contextmanager
def warnings_as_exceptions(warning_types=None, log_level=logging.WARNING):
    """
    Context manager that temporarily converts warnings and log messages to exceptions.

    This allows for tracing the origin of warnings and log messages, including TypeErrors.

    Args:
        warning_types: List of warning types to convert to exceptions. If None, all warnings are converted.
        log_level: Log level at or above which log messages are converted to exceptions.

    Usage:
        with warnings_as_exceptions([TypeError]):
            # Code that might generate warnings or log messages
            # Any specified warnings or log messages at or above the specified level will be raised as exceptions

    Raises:
        Exception: If any specified warning or log message at or above the specified level is triggered
    """
    # Set up warning filters
    with warnings.catch_warnings():
        if warning_types:
            for warning_type in warning_types:
                warnings.filterwarnings("error", category=warning_type)
        else:
            warnings.filterwarnings("error")

        # Set up logging handler
        handler = LoggingExceptionHandler(level=log_level)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        try:
            yield
        finally:
            # Clean up logging handler
            root_logger.removeHandler(handler)
