"""
Tests for the warning utilities.
"""

import logging
import warnings
from unittest.mock import patch

import pytest
from astropy.units import UnitsWarning

from exotools.utils.warning_utils import (
    LoggingExceptionHandler,
    silence_warnings,
    units_warnings_as_exceptions,
    warnings_as_exceptions,
)


class TestUnitsWarningsAsExceptions:
    """Tests for the units_warnings_as_exceptions context manager."""

    def test_raises_on_units_warning(self):
        with pytest.raises(UnitsWarning):
            with units_warnings_as_exceptions():
                warnings.warn("bad unit", UnitsWarning)

    def test_no_exception_without_warning(self):
        with units_warnings_as_exceptions():
            pass

    @patch("exotools.utils.warning_utils.HAS_UNIT_PARSER_WARNING", True)
    def test_raises_on_unit_parser_warning_when_available(self):
        from exotools.utils.warning_utils import UnitParserWarning

        with pytest.raises(UnitParserWarning):
            with units_warnings_as_exceptions():
                warnings.warn("parse fail", UnitParserWarning)

    @patch("exotools.utils.warning_utils.HAS_UNIT_PARSER_WARNING", False)
    def test_no_unit_parser_filter_when_unavailable(self):
        with units_warnings_as_exceptions():
            pass


class TestSilenceWarnings:
    """Tests for the silence_warnings context manager."""

    def test_silence_all_warnings(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with silence_warnings():
                warnings.warn("should be silenced", UserWarning)
        assert len(w) == 0

    def test_silence_specific_warning_type(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with silence_warnings(warning_types=[UnitsWarning]):
                warnings.warn("units issue", UnitsWarning)
        assert len(w) == 0

    def test_other_warnings_not_silenced(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with silence_warnings(warning_types=[UnitsWarning]):
                warnings.warn("different warning", DeprecationWarning)
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    @patch("exotools.utils.warning_utils.HAS_UNIT_PARSER_WARNING", False)
    def test_skips_unit_parser_warning_when_unavailable(self):
        from exotools.utils.warning_utils import UnitParserWarning

        with silence_warnings(warning_types=[UnitParserWarning]):
            pass


class TestLoggingExceptionHandler:
    """Tests for the LoggingExceptionHandler."""

    def test_default_level_is_warning(self):
        handler = LoggingExceptionHandler()
        assert handler.level == logging.WARNING

    def test_custom_level(self):
        handler = LoggingExceptionHandler(level=logging.ERROR)
        assert handler.level == logging.ERROR

    def test_emit_raises_exception(self):
        handler = LoggingExceptionHandler()
        logger = logging.getLogger("test_emit")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        try:
            with pytest.raises(Exception, match="Log message converted to exception"):
                logger.warning("test message")
        finally:
            logger.removeHandler(handler)


class TestWarningsAsExceptions:
    """Tests for the warnings_as_exceptions context manager."""

    def test_raises_on_any_warning_by_default(self):
        with pytest.raises(UserWarning):
            with warnings_as_exceptions():
                warnings.warn("boom", UserWarning)

    def test_raises_on_specific_warning_type(self):
        with pytest.raises(UnitsWarning):
            with warnings_as_exceptions(warning_types=[UnitsWarning]):
                warnings.warn("bad unit", UnitsWarning)

    def test_no_exception_without_warning(self):
        with warnings_as_exceptions():
            pass

    def test_raises_on_log_message(self):
        with pytest.raises(Exception, match="Log message converted to exception"):
            with warnings_as_exceptions():
                logging.warning("log warning")

    def test_custom_log_level(self):
        with pytest.raises(Exception, match="Log message converted to exception"):
            with warnings_as_exceptions(log_level=logging.ERROR):
                logging.error("log error")

    def test_cleans_up_handler_after_exception(self):
        root_logger = logging.getLogger()
        handlers_before = len(root_logger.handlers)
        with pytest.raises(Exception):
            with warnings_as_exceptions():
                logging.warning("trigger cleanup")
        assert len(root_logger.handlers) == handlers_before

    def test_cleans_up_handler_on_normal_exit(self):
        root_logger = logging.getLogger()
        handlers_before = len(root_logger.handlers)
        with warnings_as_exceptions():
            pass
        assert len(root_logger.handlers) == handlers_before

    @patch("exotools.utils.warning_utils.HAS_UNIT_PARSER_WARNING", False)
    def test_skips_unit_parser_warning_when_unavailable(self):
        from exotools.utils.warning_utils import UnitParserWarning

        with warnings_as_exceptions(warning_types=[UnitParserWarning]):
            pass
