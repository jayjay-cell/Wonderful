"""Offline-mode and logging-allowlist tests (FR-F6, FR-G5).

These verify the two mechanisms that are easy to believe are working when
they are not: a network guard that never actually blocks anything, and a
logger that quietly logs the message text it was supposed to drop.
"""

from __future__ import annotations

import logging
import socket

import pytest

from obs.logging import ALLOWED_FIELDS, StructuredLogger
from providers import offline


class TestOfflineGuard:
    """The offline guard blocks outbound connections but permits localhost."""
    def test_guard_is_active_in_the_test_suite(self) -> None:
        """conftest.py enables it for every test, so 'the core path needs
        no network' is continuously verified rather than assumed (NFR-2)."""
        assert offline.is_active()

    def test_outbound_connection_is_blocked(self) -> None:
        """A connect to a public address raises NetworkAccessBlocked."""
        with pytest.raises(offline.NetworkAccessBlocked) as err:
            socket.create_connection(("example.com", 443), timeout=1)
        assert "example.com" in str(err.value)

    def test_error_names_the_host_and_the_remedy(self) -> None:
        """The error has to be self-explanatory to someone who has never
        read providers/offline.py."""
        with pytest.raises(offline.NetworkAccessBlocked) as err:
            socket.create_connection(("api.anthropic.com", 443), timeout=1)
        message = str(err.value)
        assert "api.anthropic.com" in message
        assert "LLM_PROVIDER=ollama" in message
        assert "no internet" in message

    def test_localhost_is_permitted(self) -> None:
        """A local model server is reached on 127.0.0.1, which is exactly
        the air-gapped configuration -- blocking it would defeat the
        purpose. Connection refused is fine; being blocked is not."""
        try:
            socket.create_connection(("127.0.0.1", 11434), timeout=0.2)
        except offline.NetworkAccessBlocked:
            pytest.fail("offline mode must not block loopback connections")
        except OSError:
            pass  # nothing listening: expected and irrelevant here

    def test_install_is_idempotent(self) -> None:
        """Stacking wrappers would make the originals unrecoverable."""
        assert offline.install() is True
        assert offline.install() is True
        with pytest.raises(offline.NetworkAccessBlocked):
            socket.create_connection(("example.com", 80), timeout=1)


class TestLoggingAllowlist:
    """FR-G5: an allowlist, so a NEW field is dropped by default rather
    than logged by default. That is what makes 'we never log message text'
    structural instead of a habit."""

    def test_allowed_field_is_logged(self, caplog) -> None:
        """A field on the allowlist reaches the log line."""
        logger = StructuredLogger("test")
        with caplog.at_level(logging.INFO):
            logger.info("turn.completed", session_id="abc123", latency_ms=42)
        assert "session_id=abc123" in caplog.text
        assert "latency_ms=42" in caplog.text

    @pytest.mark.parametrize("field,value", [
        ("message_text", "the trainee said something sensitive"),
        ("content", "operator speech"),
        ("tool_args", {"fuel": 180}),
        ("briefing", "mission briefing text"),
        ("api_key", "secret-key-value"),
        ("transcript", "full conversation"),
    ])
    def test_disallowed_fields_are_dropped(self, caplog, field, value) -> None:
        """Mission content is filtered out of logs entirely."""
        logger = StructuredLogger("test")
        with caplog.at_level(logging.INFO):
            logger.info("event.name", **{field: value})
        assert field not in caplog.text
        assert str(value) not in caplog.text

    def test_allowlist_excludes_obvious_content_fields(self) -> None:
        """A direct assertion on the allowlist itself, so adding a
        content-bearing field is a deliberate act someone must justify."""
        for forbidden in ("message", "message_text", "content", "text",
                          "tool_args", "arguments", "transcript", "briefing",
                          "api_key", "token"):
            assert forbidden not in ALLOWED_FIELDS

    def test_parameter_id_allowed_but_not_its_value(self) -> None:
        """Knowing WHICH parameter changed is useful for debugging; its
        value is mission content."""
        assert "parameter_id" in ALLOWED_FIELDS
        assert "parameter_value" not in ALLOWED_FIELDS
        assert "value" not in ALLOWED_FIELDS

    def test_error_code_allowed_but_not_exception_message(self) -> None:
        """A raw exception string can carry a file path, a SQL fragment or
        user content (FR-G4)."""
        assert "error_code" in ALLOWED_FIELDS
        assert "error_message" not in ALLOWED_FIELDS
        assert "traceback" not in ALLOWED_FIELDS

    def test_event_name_always_appears(self, caplog) -> None:
        """The event name is logged even with no fields."""
        logger = StructuredLogger("test")
        with caplog.at_level(logging.INFO):
            logger.info("mission.loaded", mission_id="uav_operator_basic")
        assert "mission.loaded" in caplog.text

    def test_none_values_are_omitted(self, caplog) -> None:
        """A None field is skipped rather than logged as 'None'."""
        logger = StructuredLogger("test")
        with caplog.at_level(logging.INFO):
            logger.info("x", session_id="s1", trigger_id=None)
        assert "trigger_id" not in caplog.text
