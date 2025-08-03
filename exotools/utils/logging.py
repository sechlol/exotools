# """Logging configuration for exotools.
#
# This module follows loguru's recommended pattern for libraries:
# 1. By default, all exotools logs are disabled
# 2. Users can enable exotools logs by calling logger.enable("exotools")
# 3. Users can configure their own logger format without interference
# """
#
# import sys
# import atexit
# from IPython import get_ipython
# import logging
#
# # By default, disable all logging from the exotools library
# # Users can enable it with logger.enable("exotools") in their application
# logger.disable("exotools")
#
# # Store the handler ID to avoid duplication in Jupyter
# _JUPYTER_HANDLER_ID = None
#
#
# def setup_logger(show_file_origin: bool = False) -> None:
#     """Configure the loguru logger with exotools' preferred format.
#
#     This is intended for users of exotools who want to use our recommended
#     logging format. It's not called automatically when exotools is imported.
#
#     Args:
#         show_file_origin: Whether to show file, function and line information in logs.
#                           Default is False (hide file origin).
#     """
#     # Remove all existing handlers
#     logger.remove()
#
#     # Define the format based on whether we want to show file origin
#     if show_file_origin:
#         format_str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
#     else:
#         format_str = (
#             "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
#         )
#
#     # Add a new handler with the custom format
#     logger.configure(
#         handlers=[
#             {
#                 "sink": sys.stderr,
#                 "format": format_str,
#                 "colorize": True,
#                 "diagnose": True,  # Keep exception tracebacks
#                 "backtrace": True,  # Show traceback for caught exceptions
#             }
#         ]
#     )
#
#     # Make sure exotools logs are enabled
#     logger.enable("exotools")
#
#
# def setup_jupyter_logger(show_file_origin: bool = False) -> None:
#     """Configure the loguru logger specifically for Jupyter notebooks.
#
#     This function addresses the issue of duplicate logs in Jupyter notebooks
#     by using a special configuration that prevents handler duplication.
#
#     Args:
#         show_file_origin: Whether to show file, function and line information in logs.
#                           Default is False (hide file origin).
#     """
#     global _JUPYTER_HANDLER_ID
#
#     # First, remove all existing handlers
#     logger.remove()
#
#     # If we already have a handler, remove it to prevent duplication
#     if _JUPYTER_HANDLER_ID is not None:
#         try:
#             logger.remove(_JUPYTER_HANDLER_ID)
#         except ValueError:
#             # Handler might have been removed already
#             pass
#
#     # Define the format based on whether we want to show file origin
#     if show_file_origin:
#         format_str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
#     else:
#         format_str = (
#             "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
#         )
#
#     # Create a unique ID for this handler to prevent duplication in Jupyter
#     _JUPYTER_HANDLER_ID = logger.add(
#         sys.stderr,
#         format=format_str,
#         colorize=True,
#         diagnose=True,
#         backtrace=True,
#         enqueue=True,  # This helps with Jupyter's asynchronous execution
#     )
#
#     # Make sure exotools logs are enabled
#     logger.enable("exotools")
#
#     # Register a cleanup function to remove the handler when the kernel shuts down
#     if is_jupyter():
#         try:
#             get_ipython().events.register("pre_shutdown", lambda: logger.remove(_JUPYTER_HANDLER_ID))
#         except (AttributeError, NameError):
#             # Fall back to atexit if IPython events are not available
#             atexit.register(lambda: logger.remove(_JUPYTER_HANDLER_ID))
#
#     return _JUPYTER_HANDLER_ID
#
#
# def is_jupyter() -> bool:
#     """Determine if the code is running in a Jupyter notebook."""
#     try:
#         # Check for IPython
#         shell = get_ipython().__class__.__name__
#         if shell == "ZMQInteractiveShell":  # Jupyter notebook or qtconsole
#             return True
#         elif shell == "TerminalInteractiveShell":  # Terminal IPython
#             return False
#         else:
#             return False
#     except (NameError, AttributeError):  # Not IPython/Jupyter
#         return False
