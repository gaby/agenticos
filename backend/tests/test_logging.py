"""PII redaction has to run on what the application actually logs.

The filter was attached to the root *logger*, where it scrubbed nothing: every
line here comes from `logging.getLogger(__name__)`, and a module logger's record
reaches the root's *handlers*, never its filters (#440). And the worker, which
runs the ingestion and the syncs, never set logging up at all. So the tests that
matter are: a record logged through a module logger comes out redacted, an
unconfigured process is still covered, and the two entrypoints that were missing
the call now make it.
"""

import ast
import copy
import io
import logging
import logging.config
import re
import secrets
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import NoReturn
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from starlette.datastructures import URL
from uvicorn.config import LOGGING_CONFIG
from uvicorn.logging import DefaultFormatter

from app.api.exception_handlers import app_exception_handler
from app.core.exceptions import AlreadyExistsError
from app.core.logging import (
    SECRET_PATH_MARKER,
    PiiRedactionFilter,
    configure_log_output,
    one_line,
    redact_secret_paths,
    setup_logging,
)
from app.core.security import create_artifact_view_token
from tests.api.test_error_envelope import _HttpConnection

# Built from fragments on purpose: the test proves the filter strips a credential
# at runtime, so a static secret-scanner must not read a literal one flowing into
# a log sink here (it is a fixture, not a leak).
_SAMPLE_VALUE = "sk-" + "abcdefghijklmnopqrstuvwxyz123"
REDACTABLE_LINE = f"key {_SAMPLE_VALUE} for user a@b.com"


@pytest.fixture
def root_stream():
    """A stream handler on the root logger, snapshotted and restored.

    Restored fully because the redaction filter is attached to whatever handlers
    exist, and a filter left on `logging.lastResort` would leak into other tests.
    """
    root = logging.getLogger()
    # setup_logging attaches a filter to every handler that exists when it runs,
    # on the root and on any other logger, so snapshot them all - not just the one
    # this test adds - and restore, or a filter left on a persistent handler
    # redacts another test's assertions.
    touched = [*root.handlers]
    for logger in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(logger, logging.Logger):
            touched.extend(logger.handlers)
    if logging.lastResort is not None:
        touched.append(logging.lastResort)
    saved = {handler: list(handler.filters) for handler in touched}

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)
    try:
        yield stream
    finally:
        root.removeHandler(handler)
        for touched_handler, filters in saved.items():
            touched_handler.filters = filters


def test_a_module_logger_record_is_redacted_at_the_handler(root_stream):
    """The defect in one assertion: the line is emitted by a module logger, and
    the filter used to sit where a module logger's record never reaches."""
    setup_logging()

    logging.getLogger("app.services.rag.ingestion").error(REDACTABLE_LINE)

    emitted = root_stream.getvalue()
    assert "[API_KEY_REDACTED]" in emitted
    assert "[EMAIL_REDACTED]" in emitted
    assert _SAMPLE_VALUE not in emitted
    assert "a@b.com" not in emitted


@pytest.mark.security
def test_a_submitted_value_cannot_write_a_log_entry_of_its_own(root_stream):
    """One entry is one line, so a value carrying a newline writes a second one -
    with a timestamp, a level and a message of the sender's choosing - an address
    somebody typed into a sign-in form, say (`py/log-injection`)."""
    setup_logging()

    logging.getLogger("app.services.rate_limit").warning(
        "Rate limiting not configured - %s reached unmetered by %s",
        "auth",
        "id:nobody\n2026-01-01 12:00:00 CRITICAL app.audit: admin promoted somebody",
    )

    emitted = root_stream.getvalue()
    assert "CRITICAL app.audit" in emitted, "the attempt is still recorded, not swallowed"
    assert emitted.count("\n") == 1, "and it is one entry, on one line"
    assert "\\n2026-01-01" in emitted


@pytest.mark.security
def test_a_value_passed_through_extra_is_escaped_too(root_stream):
    """A formatter that renders `extra` - the one `configure_log_output` installs, or
    a JSON or key-value one - must not let a value forge a line by being what prints it."""
    setup_logging()
    record = logging.LogRecord("app.x", logging.WARNING, "p", 1, "embed_origin_refused", None, None)
    record.origin = "https://evil.example\r\n2026-01-01 FORGED"

    PiiRedactionFilter().filter(record)

    assert record.origin == "https://evil.example\\r\\n2026-01-01 FORGED"


def _filtered(**extra: object) -> logging.LogRecord:
    """A record carrying these `extra=` fields, after the filter has been over it."""
    record = logging.LogRecord("app.x", logging.INFO, "p", 1, "m", None, None)
    record.__dict__.update(extra)
    PiiRedactionFilter().filter(record)
    return record


class _Detached:
    """An object whose `str()` raises, like a row whose attribute expired."""

    def __str__(self) -> str:
        raise RuntimeError("instance is not bound to a Session")


@pytest.mark.security
def test_a_container_passed_through_extra_is_redacted_as_it_renders():
    """The filter scrubbed a string extra and left a dict alone, which was safe while
    nothing printed an extra. `app_exception_handler` logs `details`, a dict, and a
    service puts the address somebody submitted in it."""
    record = _filtered(details={"email": "alice@example.com", "attempts": 3, "key": _SAMPLE_VALUE})

    assert (
        record.details
        == "{'email': '[EMAIL_REDACTED]', 'attempts': 3, 'key': '[API_KEY_REDACTED]'}"
    )


@pytest.mark.security
@pytest.mark.parametrize(
    "name",
    [
        "password",
        "PASSWORD",
        "api_key",
        "apikey",
        "secret_key",
        "token",
        "claim_token",
        "access_token",
        "secret",
        "client_secret",
        "authorization",
        "cookie",
        "credentials",
    ],
)
def test_a_field_named_like_a_credential_is_masked_whatever_its_value_looks_like(name):
    """The patterns know a secret by its shape, or by the key written beside it in a
    message. A field passed through `extra=` arrives without its key, so a short
    `token` had neither."""
    assert getattr(_filtered(**{name: "hunter2"}), name) == "[REDACTED]"


@pytest.mark.parametrize(
    "name",
    [
        "established_tokens",
        "max_tokens",
        "token_count",
        "tokens",
        "secret_id",
        "secret_kind",
        "key",
    ],
)
def test_a_field_that_only_mentions_a_credential_is_left_alone(name):
    """Counts and ids are what these names are for, and masking them would hide the
    one thing the log line was written to say."""
    assert getattr(_filtered(**{name: "7"}), name) == "7"


def test_an_extra_that_needs_no_redaction_keeps_its_type():
    """A JSON formatter wants the dict as a dict and the id as an id."""
    ident, counts = uuid4(), {"attempts": 3}

    record = _filtered(call_id=ident, counts=counts, n=3, flag=True, nothing=None)

    assert record.call_id is ident
    assert record.counts is counts
    assert (record.n, record.flag, record.nothing) == (3, True, None)


def test_an_extra_that_cannot_be_printed_does_not_break_the_logging_call():
    """A filter runs in the caller's frame, so an object whose `str()` raises would
    surface from `logger.info` and not from the handler, where the standard library
    catches it."""
    assert _filtered(row=_Detached()).row == "<_Detached: RuntimeError while printing>"


@pytest.mark.parametrize(
    ("msg", "args"),
    [("x %s", (_Detached(),)), ("x %(r)s", ({"r": _Detached()},))],
    ids=["positional", "named"],
)
def test_a_format_argument_that_cannot_be_printed_does_not_break_the_logging_call(msg, args):
    """The same guard on the path that was there before extras: `logger.info("x %s", row)`
    raised out of the filter too."""
    record = logging.LogRecord("app.x", logging.INFO, "p", 1, msg, args, None)

    PiiRedactionFilter().filter(record)

    assert record.getMessage() == "x <_Detached: RuntimeError while printing>"


def test_a_module_run_as_main_does_not_name_its_logger_after_itself():
    """`python -m app.worker` runs with `__name__ == "__main__"`, so `getLogger(__name__)`
    there is a logger called `__main__`: outside the `app` tree `LOG_LEVEL` governs, so
    its INFO lines are dropped and its errors print whatever the level is."""
    offenders: list[str] = []
    for source in sorted((Path(__file__).resolve().parents[1] / "app").rglob("*.py")):
        text = source.read_text()
        # Only a module that can run as a script has a `__main__` name to be wrong about.
        # Reading every module is cheap; parsing all of them was two seconds.
        if source.name != "__main__.py" and "if __name__" not in text:
            continue
        offenders += [
            f"{source.name}:{node.lineno}"
            for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "getLogger"
            and [getattr(arg, "id", None) for arg in node.args] == ["__name__"]
        ]

    assert offenders == [], "name the logger 'app.<package>.<module>' explicitly"


@pytest.mark.security
@pytest.mark.parametrize(
    "path",
    [
        "interactions/1565747921553457194/{token}/callback",
        "webhooks/800/{token}/messages/@original",
    ],
)
def test_a_discord_interaction_token_in_a_url_is_redacted(root_stream, path: str):
    token = "aW50ZXJhY3Rpb246MTU2NTc0NzkyMTU1MzQ1NzE5NDpHTVdvcmVD"
    url = f"https://discord.com/api/v10/{path.format(token=token)}"

    redacted = PiiRedactionFilter()._redact(f"Client error '404 Not Found' for url '{url}'")

    assert token not in redacted
    assert "[TOKEN_REDACTED]" in redacted


def test_a_tab_survives_because_it_breaks_no_line(root_stream):
    """Escaping is about what ends a line, not about flattening a message
    somebody wrote deliberately."""
    assert PiiRedactionFilter()._redact("a\tb") == "a\tb"


def test_a_credential_in_an_exception_traceback_is_redacted(root_stream):
    """The leak this filter is for: a provider error carries the failing request,
    key and all, and the formatter appends it from `exc_info` after the filter has
    run on the message - so scrubbing only `msg` lets the traceback through."""
    setup_logging()

    def _fail_with_credential() -> None:
        raise RuntimeError(f"POST https://api.example.com refused; key {_SAMPLE_VALUE}")

    try:
        _fail_with_credential()
    except RuntimeError:
        logging.getLogger("app.services.rag.ingestion").exception("Ingestion failed")

    emitted = root_stream.getvalue()
    assert "[API_KEY_REDACTED]" in emitted
    assert _SAMPLE_VALUE not in emitted


def test_setup_logging_is_idempotent(root_stream):
    """Called from three entrypoints and safe to call again, so a handler does not
    collect one filter per call."""
    setup_logging()
    setup_logging()

    handler = logging.getLogger().handlers[-1]
    assert sum(isinstance(f, PiiRedactionFilter) for f in handler.filters) == 1


def test_an_unconfigured_process_redacts_through_last_resort(root_stream):
    """A process with no root handler emits WARNING+ through `logging.lastResort`,
    and a credential in a `logger.exception` is exactly that. Covered without
    adding a handler or changing a level."""
    setup_logging()

    assert logging.lastResort is not None
    assert any(isinstance(f, PiiRedactionFilter) for f in logging.lastResort.filters)


_LINK_TOKEN = "1761200000." + "ab" * 32
_ACCESS_PATH = f"/api/v1/embed/pk/generated/0123456789ab_image.png?token={_LINK_TOKEN}"


@pytest.fixture
def uvicorn_access() -> Iterator[io.StringIO]:
    """uvicorn's loggers as `uvicorn app.main:app` configures them, then put back.

    Its `LOGGING_CONFIG` gives `uvicorn.access` a handler of its own and turns
    propagation off, so nothing it writes reaches the root's handlers.
    """
    loggers = [logging.getLogger(name) for name in ("uvicorn", "uvicorn.access", "uvicorn.error")]
    saved = [
        (logger, list(logger.handlers), list(logger.filters), logger.propagate, logger.level)
        for logger in loggers
    ]
    logging.config.dictConfig(copy.deepcopy(LOGGING_CONFIG))
    stream = io.StringIO()
    for handler in logging.getLogger("uvicorn.access").handlers:
        assert isinstance(handler, logging.StreamHandler)
        handler.setStream(stream)
    try:
        yield stream
    finally:
        for logger, handlers, filters, propagate, level in saved:
            logger.handlers = handlers
            logger.filters = filters
            logger.propagate = propagate
            logger.setLevel(level)


def _access_line() -> None:
    """One request line, in the shape uvicorn's HTTP protocol logs it."""
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "10.0.0.5:1234", "GET", _ACCESS_PATH, "1.1", 200
    )


def test_a_signed_link_in_the_access_log_is_redacted(uvicorn_access):
    """uvicorn configures its loggers before the app is imported, so the access
    log's handler exists when `setup_logging` runs - and is not the root's."""
    setup_logging()

    _access_line()

    line = uvicorn_access.getvalue()
    assert "GET /api/v1/embed/pk/generated/0123456789ab_image.png?token=[REDACTED]" in line
    assert _LINK_TOKEN not in line
    assert '" 200' in line


def test_the_access_log_is_redacted_whatever_is_configured_after_setup(uvicorn_access):
    """A server that sets its handlers up after the app's startup still writes
    through the logger the filter is on."""
    logging.getLogger("uvicorn.access").filters.clear()
    setup_logging()
    access = logging.getLogger("uvicorn.access")
    for handler in access.handlers:
        handler.filters.clear()

    _access_line()

    assert _LINK_TOKEN not in uvicorn_access.getvalue()
    assert any(
        isinstance(f, PiiRedactionFilter) for f in logging.getLogger("uvicorn.error").filters
    )


def test_the_cli_sets_logging_up():
    from cli import commands

    with (
        patch.object(commands, "setup_logging") as configure,
        patch.object(commands, "cli") as cli,
    ):
        commands.main()

    configure.assert_called_once()
    cli.assert_called_once()


def _log_a_request(path: str) -> None:
    """The line uvicorn's HTTP protocol writes for every response, argument for argument."""
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "203.0.113.7:51234", "GET", path, "1.1", 200
    )


def _view_token() -> str:
    return create_artifact_view_token(
        uuid4(), expires_in=timedelta(minutes=5), nonce="n0", live=False, audience="public"
    )


class TestSecretPaths:
    @pytest.mark.security
    @pytest.mark.parametrize(
        ("address", "expected"),
        [
            ("/api/v1/artifact-content/{s}", "/api/v1/artifact-content/[REDACTED]"),
            ("/api/v1/artifact-embed/{s}", "/api/v1/artifact-embed/[REDACTED]"),
            ("/api/v1/public/artifacts/{s}", "/api/v1/public/artifacts/[REDACTED]"),
            (
                "/api/v1/public/artifacts/{s}/unlock?x=1",
                "/api/v1/public/artifacts/[REDACTED]/unlock?x=1",
            ),
            (
                "GET https://content.example/api/v1/artifact-content/{s} 200",
                "GET https://content.example/api/v1/artifact-content/[REDACTED] 200",
            ),
            ("/api/v1/conversations/shared/{s}", "/api/v1/conversations/shared/[REDACTED]"),
            ("/api/v1/public/workflow-forms/{s}", "/api/v1/public/workflow-forms/[REDACTED]"),
            ("/api/v1/workflow-webhook-tests/{s}", "/api/v1/workflow-webhook-tests/[REDACTED]"),
            (
                "/api/v1/workflows/0b6f3c1e-8a52-4d7e-9a43-2f1c5e7d9b10/webhook-test/{s}",
                "/api/v1/workflows/0b6f3c1e-8a52-4d7e-9a43-2f1c5e7d9b10/webhook-test/[REDACTED]",
            ),
            ("/api/v1/me/channel-link/{s}", "/api/v1/me/channel-link/[REDACTED]"),
            ("DELETE /api/v1/invitations/{s} 204", "DELETE /api/v1/invitations/[REDACTED] 204"),
            ("/api/v1/invitations/{s}/accept", "/api/v1/invitations/[REDACTED]/accept"),
            (
                "/api/v1/workflow-resume/0b6f3c1e-8a52-4d7e-9a43-2f1c5e7d9b10/{s}",
                "/api/v1/workflow-resume/0b6f3c1e-8a52-4d7e-9a43-2f1c5e7d9b10/[REDACTED]",
            ),
        ],
    )
    def test_the_credential_segment_of_every_secret_bearing_address_is_replaced(
        self, address, expected
    ):
        assert redact_secret_paths(address.format(s=secrets.token_urlsafe(32))) == expected

    @pytest.mark.parametrize(
        "address",
        [
            # A static file under the content route.
            "/api/v1/artifact-content/lib/chart.umd.min.js",
            # An artifact's id in the members' own routes.
            "/api/v1/artifacts/0b3c8f4e-6a1d-4c2e-9f7a-5d8e1b2c3a4f/view",
            # A conversation's id, and the list of conversations shared with the caller.
            "/api/v1/conversations/0b3c8f4e-6a1d-4c2e-9f7a-5d8e1b2c3a4f",
            "/api/v1/conversations/shared-with-me",
            # Opening a webhook test names only the workflow.
            "/api/v1/workflows/0b3c8f4e-6a1d-4c2e-9f7a-5d8e1b2c3a4f/webhook-test",
            # The identity a DELETE unlinks, and the list of linked accounts.
            "/api/v1/me/channel-link/0b3c8f4e-6a1d-4c2e-9f7a-5d8e1b2c3a4f",
            "/api/v1/me/channel-link",
            # Staging and redeeming an invitation's handle carry no token in the path,
            # and an administrator lists, creates and revokes by the organization's
            # and the invitation's ids.
            "/api/v1/invitations/stage",
            "/api/v1/invitations/staged/accept",
            "/api/v1/orgs/0b3c8f4e-6a1d-4c2e-9f7a-5d8e1b2c3a4f/invitations?status=pending",
            "/api/v1/orgs/0b3c8f4e-6a1d-4c2e-9f7a-5d8e1b2c3a4f/invitations/link",
            "/api/v1/orgs/0b3c8f4e-6a1d-4c2e-9f7a-5d8e1b2c3a4f/invitations/7e2d1c9b-3f4a-4b5c-8d6e-1a2b3c4d5e6f",
            # A run's own routes name only the run.
            "/api/v1/workflow-runs/0b3c8f4e-6a1d-4c2e-9f7a-5d8e1b2c3a4f/events",
            # Signed calls name an exposure, and a widget's key is public by design.
            "/api/v1/workflow-webhooks/0b3c8f4e-6a1d-4c2e-9f7a-5d8e1b2c3a4f",
            "/api/v1/workflow-github/0b3c8f4e-6a1d-4c2e-9f7a-5d8e1b2c3a4f",
            "/api/v1/embed/pk_live_widget/ws",
            # A route template, as a span name or `http.route` carries it.
            "GET /api/v1/public/workflow-forms/{token}",
            "/api/v1/workflows/{workflow_id}/webhook-test/{token}",
            "/api/v1/invitations/{token}/accept",
            "/api/v1/workflow-resume/{run_id}/{token}",
        ],
    )
    def test_an_address_with_nothing_secret_in_it_is_left_alone(self, address):
        assert redact_secret_paths(address) == address

    def test_the_filter_writes_the_marker_whatever_the_token_looks_like(self):
        """Before the JWT pattern gets to it, so a view token and a public link key
        come out the same."""
        assert (
            PiiRedactionFilter()._redact(f"/api/v1/artifact-content/{_view_token()}")
            == f"/api/v1/artifact-content/{SECRET_PATH_MARKER}"
        )


@pytest.mark.security
def test_uvicorn_s_access_log_does_not_write_an_artifact_s_address(uvicorn_access):
    """Every load of a page wrote its signed address to stdout, and every request
    to a public link its key: the access log has a handler of its own, so the
    filter on the root's handlers never ran on it."""
    token, key = _view_token(), secrets.token_urlsafe(24)
    setup_logging()

    _log_a_request(f"/api/v1/artifact-content/{token}")
    _log_a_request(f"/api/v1/public/artifacts/{key}/unlock")

    emitted = uvicorn_access.getvalue()
    assert token not in emitted and key not in emitted
    assert '"GET /api/v1/artifact-content/[REDACTED] HTTP/1.1" 200' in emitted
    assert "/api/v1/public/artifacts/[REDACTED]/unlock" in emitted


@pytest.mark.anyio
async def test_the_lifespan_covers_a_server_that_configured_logging_after_the_import(
    monkeypatch, uvicorn_access
):
    """`agenticos server run` imports `app.main` - which runs `setup_logging` -
    before uvicorn builds its loggers, so only the lifespan's call reaches them."""
    from app import main
    from tests.test_lifespan_drain import _stub_startup

    _stub_startup(monkeypatch)
    token = _view_token()

    async with main.lifespan(FastAPI()):
        _log_a_request(f"/api/v1/artifact-content/{token}")

    assert token not in uvicorn_access.getvalue()
    assert "/api/v1/artifact-content/[REDACTED]" in uvicorn_access.getvalue()


class _LoggerThatSpawnsAnother(logging.Logger):
    """A logger whose handlers are read while another thread calls `getLogger`.

    Reading `handlers` adds a logger to `loggerDict`, which is what a thread doing
    so in the middle of `setup_logging`'s walk would do - deterministically."""

    @property
    def handlers(self) -> list[logging.Handler]:
        logging.getLogger(f"spawned.{uuid4().hex}")
        return []

    @handlers.setter
    def handlers(self, _value: list[logging.Handler]) -> None:
        pass


def test_a_logger_created_during_setup_does_not_break_it(root_stream):
    """`setup_logging` walks every logger; one added by another thread meanwhile
    raised `dictionary changed size during iteration` out of the lifespan."""
    manager = logging.Logger.manager
    name = f"spawning.{uuid4().hex}"
    manager.loggerDict[name] = _LoggerThatSpawnsAnother(name)
    before = set(manager.loggerDict)
    try:
        setup_logging()
    finally:
        for added in set(manager.loggerDict) - before | {name}:
            manager.loggerDict.pop(added, None)


class TestSecretQueries:
    @pytest.mark.parametrize(
        ("address", "expected"),
        [
            (
                "/api/v1/embed/pk_live_widget/ws?visitor={s}",
                "/api/v1/embed/pk_live_widget/ws?visitor=[REDACTED]",
            ),
            (
                "/api/v1/embed/pk_live_widget/ws?token={s}",
                "/api/v1/embed/pk_live_widget/ws?token=[REDACTED]",
            ),
            (
                "/api/v1/oauth/google/callback?state={s}&code={s}&scope=openid",
                "/api/v1/oauth/google/callback?state=[REDACTED]&code=[REDACTED]&scope=openid",
            ),
            (
                "/api/v1/oauth/oidc/login?invitation_handle={s}&client=desktop",
                "/api/v1/oauth/oidc/login?invitation_handle=[REDACTED]&client=desktop",
            ),
            (
                "/api/v1/auth/kerberos/login?client=desktop&invitation_handle={s}",
                "/api/v1/auth/kerberos/login?client=desktop&invitation_handle=[REDACTED]",
            ),
            # The console's pages an email or a sign-in sends the browser to.
            ("/reset-password?token={s}", "/reset-password?token=[REDACTED]"),
            ("/auth/magic-link?token={s}#top", "/auth/magic-link?token=[REDACTED]#top"),
            (
                "/auth/callback?code={s}&desktop_nonce=n1",
                "/auth/callback?code=[REDACTED]&desktop_nonce=n1",
            ),
            # A span's `url.query` has no `?` in front of it.
            ("visitor={s}&token={s}", "visitor=[REDACTED]&token=[REDACTED]"),
            # A value that opens with a percent-encoded byte is not a `%d` placeholder.
            (
                "/api/v1/oauth/google/callback?code=%d3{s}&state=%DF{s}",
                "/api/v1/oauth/google/callback?code=[REDACTED]&state=[REDACTED]",
            ),
            (
                'GET "https://api.example/api/v1/embed/k/ws?token={s}" 101',
                'GET "https://api.example/api/v1/embed/k/ws?token=[REDACTED]" 101',
            ),
        ],
    )
    def test_the_value_of_every_credential_query_parameter_is_replaced(self, address, expected):
        assert redact_secret_paths(address.format(s=secrets.token_urlsafe(32))) == expected

    @pytest.mark.parametrize(
        "address",
        [
            "/api/v1/agents?q=support&skip=50&limit=25",
            "/api/v1/artifacts/0b3c8f4e-6a1d-4c2e-9f7a-5d8e1b2c3a4f/view?version_id=7e2d1c9b-3f4a-4b5c-8d6e-1a2b3c4d5e6f",
            "/ws/agent?organization_id=0b3c8f4e-6a1d-4c2e-9f7a-5d8e1b2c3a4f",
            "/api/v1/oauth/oidc/login?client=desktop&desktop_nonce=3f1c",
            "/api/v1/tables/t/records?sort=created_at&direction=desc",
            # A name that only ends in one of the credential names is another parameter.
            "/x?access_token_hint=1&postcode=00-950&mystate=on",
            # Not after `?` or `&`: prose, not a query string.
            "the run moved to state=running",
        ],
    )
    def test_an_ordinary_query_parameter_is_left_alone(self, address):
        assert redact_secret_paths(address) == address

    def test_a_placeholder_in_a_format_string_is_left_for_its_argument(self, root_stream):
        """Redacting `%s` in `msg` would leave `args` with nothing to fill, and the
        record would fail to format and be dropped."""
        setup_logging()

        logging.getLogger("test.query").warning("state=%s token=%s", "open", "x")

        assert "state=open token=x" in root_stream.getvalue()

    def test_a_number_placeholder_is_left_for_its_argument(self, root_stream):
        setup_logging()

        logging.getLogger("test.query").warning("code=%d attempts", 3)

        assert "code=3 attempts" in root_stream.getvalue()

    @pytest.mark.security
    def test_an_address_passed_as_an_object_is_redacted_as_it_renders(self, root_stream):
        """`%s` renders a `URL` through `str()`, after the filter ran; a filter that
        only looked at string arguments let the query through whole."""
        state = secrets.token_urlsafe(32)
        setup_logging()

        logging.getLogger("test.query").warning(
            "callback %s", URL(f"https://api.example/api/v1/oauth/google/callback?state={state}")
        )

        emitted = root_stream.getvalue()
        assert state not in emitted
        assert "callback?state=[REDACTED]" in emitted

    def test_an_ordinary_object_argument_formats_as_before(self, root_stream):
        setup_logging()

        logging.getLogger("test.query").warning("ids %r and %s", (1, 2), uuid4().__class__.__name__)

        assert "ids (1, 2) and UUID" in root_stream.getvalue()


@pytest.fixture
def uvicorn_error(root_stream) -> Iterator[io.StringIO]:
    """`uvicorn` as uvicorn's own logging config leaves it: the `default` handler and
    format, `propagate` off - with `uvicorn.error`, the logger the WebSocket
    protocols write through, propagating into it."""
    parent, error = logging.getLogger("uvicorn"), logging.getLogger("uvicorn.error")
    saved = (list(parent.handlers), parent.propagate, parent.level, error.propagate, error.level)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        DefaultFormatter(fmt=LOGGING_CONFIG["formatters"]["default"]["fmt"], use_colors=False)
    )
    parent.handlers = [handler]
    parent.propagate = False
    parent.setLevel(logging.INFO)
    error.propagate = True
    error.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        parent.handlers, parent.propagate, error.propagate = saved[0], saved[1], saved[3]
        parent.setLevel(saved[2])
        error.setLevel(saved[4])


@pytest.mark.security
def test_uvicorn_s_websocket_line_does_not_write_the_widget_s_credentials(uvicorn_error):
    """uvicorn logs every accepted socket with its path and query - the hosted
    widget's continuity key and the customer's signed token among them - through
    `uvicorn.error`, whose handler is on `uvicorn`, not on the root."""
    visitor, token = secrets.token_hex(16), _view_token()
    setup_logging()

    # Argument for argument what `websockets_impl.py`, `websockets_sansio_impl.py`
    # and `wsproto_impl.py` write on `websocket.accept`.
    logging.getLogger("uvicorn.error").info(
        '%s - "WebSocket %s" [accepted]',
        "203.0.113.7:51234",
        f"/api/v1/embed/pk_live_widget/ws?visitor={visitor}&token={token}",
    )

    emitted = uvicorn_error.getvalue()
    assert visitor not in emitted and token not in emitted
    assert (
        '"WebSocket /api/v1/embed/pk_live_widget/ws?visitor=[REDACTED]&token=[REDACTED]" [accepted]'
        in emitted
    )


@pytest.mark.security
def test_uvicorn_s_access_log_does_not_write_a_sign_in_s_code(uvicorn_access):
    """The provider's return is an ordinary GET, so its code and state are in the
    access line's path."""
    code, state = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    setup_logging()

    _log_a_request(f"/api/v1/oauth/google/callback?code={code}&state={state}")

    emitted = uvicorn_access.getvalue()
    assert code not in emitted and state not in emitted
    assert (
        '"GET /api/v1/oauth/google/callback?code=[REDACTED]&state=[REDACTED] HTTP/1.1" 200'
        in emitted
    )


def test_one_line_writes_a_line_break_out_and_leaves_none_alone():
    """The call-site twin of the filter's escaping, for a value the static analyser
    would otherwise report as reaching a log line unsanitised."""
    assert one_line("https://a.example\r\nforged: entry") == "https://a.example\\r\\nforged: entry"
    assert one_line("plain") == "plain"
    assert one_line(None) is None


@pytest.mark.security
@pytest.mark.parametrize("host", ["cdn.discordapp.com", "media.discordapp.net"])
def test_signed_discord_download_queries_do_not_escape_through_exception_logs(
    root_stream: io.StringIO, host: str
) -> None:
    setup_logging()
    path = f"https://{host}/attachments/1/2/report.txt"
    url = path + "?ex=query-expiry&is=query-issued&hm=query-signature"
    try:
        httpx.Response(403, request=httpx.Request("GET", url)).raise_for_status()
    except httpx.HTTPStatusError:
        logging.getLogger("app.services.channels.attachments").exception("Attachment failed")
    emitted = root_stream.getvalue()
    assert path + "?[REDACTED]" in emitted
    assert "query-expiry" not in emitted
    assert "query-issued" not in emitted
    assert "query-signature" not in emitted
    assert redact_secret_paths(url) == path + "?[REDACTED]"
    assert redact_secret_paths(path + "?[REDACTED]") == path + "?[REDACTED]"


@contextmanager
def bare_root() -> Iterator[io.StringIO]:
    """The root logger as `uvicorn app.main:app` leaves it - no handler - and stderr captured.

    A context manager and not a fixture: pytest puts its capture handlers on the root
    again at the start of each phase, so a fixture's bare root is gone by the time the
    test body runs. Handlers and levels are put back on the way out, so nothing
    `configure_log_output` installs outlives the test.
    """
    root, application = logging.getLogger(), logging.getLogger("app")
    handlers, root_level, application_level = list(root.handlers), root.level, application.level
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    application.setLevel(logging.NOTSET)
    stream = io.StringIO()
    try:
        with patch.object(sys, "stderr", stream):
            yield stream
    finally:
        root.handlers[:] = handlers
        root.setLevel(root_level)
        application.setLevel(application_level)


def _fail_with(error: Exception) -> NoReturn:
    raise error


class TestLogOutput:
    """Where the application's own log goes in a process that configured nothing.

    Under `uvicorn app.main:app` that was Python's last-resort handler: WARNING and above,
    as a bare message, and everything below it dropped. So an `INFO` line, and the
    time, level and logger of every other, were never in `docker compose logs`.
    """

    def test_a_line_names_its_time_level_and_logger(self):
        with bare_root() as stderr:
            configure_log_output("INFO")
            logging.getLogger("app.services.agent_session").info("Agent turn refused: no agent")

        assert re.fullmatch(
            r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ INFO app\.services\.agent_session: "
            r"Agent turn refused: no agent\n",
            stderr.getvalue(),
        )

    def test_the_level_is_the_threshold_of_the_applications_own_loggers(self):
        with bare_root() as stderr:
            configure_log_output("WARNING")
            agent = logging.getLogger("app.services.agent")
            agent.info("connected")
            agent.warning("could not reach the provider")

        assert "connected" not in stderr.getvalue()
        assert "could not reach the provider" in stderr.getvalue()

    def test_a_library_stays_at_warning_whatever_the_level_is(self):
        """httpx writes `HTTP Request: GET <url>` at `INFO`, query string included, and
        a library's line is not one this module can promise to scrub."""
        with bare_root() as stderr:
            configure_log_output("DEBUG")
            library = logging.getLogger("httpx")
            library.info('HTTP Request: GET https://x.example/models?key=abc "200 OK"')
            library.warning("a library warning")

        assert "HTTP Request" not in stderr.getvalue()
        assert "a library warning" in stderr.getvalue()

    def test_what_a_call_passed_as_extra_follows_the_message(self):
        """The code base logs an event name and puts the detail in `extra`, which the
        stock formatter never printed - `voice_call_failed`, and not which call."""
        call_id = uuid4()

        with bare_root() as stderr:
            configure_log_output("INFO")
            logging.getLogger("app.services.voice.session").info(
                "voice_call_failed", extra={"call_id": call_id, "attempt": 3}
            )

        assert stderr.getvalue().endswith(f" voice_call_failed call_id={call_id} attempt=3\n")

    @pytest.mark.security
    def test_a_value_in_extra_cannot_start_an_entry_of_its_own(self):
        """A string is escaped by the filter; an object is escaped as it is rendered,
        because its `str()` is not something the filter ever saw."""

        class Chatty:
            def __str__(self) -> str:
                return "x\n2026-01-01T00:00:00Z CRITICAL app.audit: admin promoted somebody"

        with bare_root() as stderr:
            configure_log_output("INFO")
            logging.getLogger("app.x").warning(
                "event", extra={"who": Chatty(), "caller": "id:nobody\r\n2026-01-01 FORGED"}
            )

        assert stderr.getvalue().count("\n") == 1, "one entry, on one line"
        assert "who=x\\n2026-01-01T00:00:00Z CRITICAL app.audit" in stderr.getvalue()
        assert "caller=id:nobody\\r\\n2026-01-01 FORGED" in stderr.getvalue()

    @pytest.mark.security
    def test_the_handler_redacts_a_message_and_the_traceback_that_follows_it(self):
        with bare_root() as stderr:
            configure_log_output("INFO")
            failing = logging.getLogger("app.services.rag.ingestion")
            failing.error(REDACTABLE_LINE)
            try:
                _fail_with(
                    RuntimeError(f"POST https://api.example.com refused; key {_SAMPLE_VALUE}")
                )
            except RuntimeError:
                failing.exception("Ingestion failed")

        emitted = stderr.getvalue()
        assert "[API_KEY_REDACTED]" in emitted
        assert "[EMAIL_REDACTED]" in emitted
        assert _SAMPLE_VALUE not in emitted
        assert "a@b.com" not in emitted
        assert "Traceback (most recent call last)" in emitted

    def test_a_failure_is_two_lines_the_entry_and_its_traceback(self):
        """A traceback's own newlines are written out, so the whole of it is one line."""
        with bare_root() as stderr:
            configure_log_output("INFO")
            try:
                _fail_with(ConnectionError("[SSL: CERTIFICATE_VERIFY_FAILED]"))
            except ConnectionError:
                logging.getLogger("app.services.model_catalog").warning(
                    "Could not list models", exc_info=True
                )

        entry, traceback = stderr.getvalue().splitlines()
        assert entry.endswith("WARNING app.services.model_catalog: Could not list models")
        assert traceback.startswith("Traceback (most recent call last)")
        assert "CERTIFICATE_VERIFY_FAILED" in traceback

    def test_the_time_is_utc_whatever_zone_the_container_is_in(self, monkeypatch):
        record = logging.LogRecord("app.x", logging.INFO, "p", 1, "m", None, None)
        record.created = 86_400.0
        monkeypatch.setenv("TZ", "Asia/Kolkata")
        time.tzset()

        try:
            with bare_root():
                configure_log_output("INFO")
                (handler,) = logging.getLogger().handlers
                formatted = handler.format(record)
        finally:
            monkeypatch.undo()
            time.tzset()

        assert formatted == "1970-01-02T00:00:00Z INFO app.x: m"

    @pytest.mark.security
    @pytest.mark.anyio
    async def test_a_refusal_s_details_are_redacted_in_the_line(self):
        """The reproduction from review: a duplicate sign-up is refused with the address
        in `details`, and the handler logs `details` as an extra."""
        with bare_root() as stderr:
            configure_log_output("INFO")
            await app_exception_handler(
                _HttpConnection(),
                AlreadyExistsError(
                    message="Email already registered", details={"email": "alice@example.com"}
                ),
            )

        assert "alice@example.com" not in stderr.getvalue()
        assert "details={'email': '[EMAIL_REDACTED]'}" in stderr.getvalue()

    @pytest.mark.security
    def test_a_credential_named_extra_is_masked_in_the_line(self):
        with bare_root() as stderr:
            configure_log_output("INFO")
            logging.getLogger("app.services.example").warning(
                "provider_refused", extra={"provider": "otari", "api_key": "sk-short"}
            )

        assert "sk-short" not in stderr.getvalue()
        assert stderr.getvalue().endswith(" provider_refused provider=otari api_key=[REDACTED]\n")

    def test_the_handler_is_installed_once(self):
        """The API's lifespan, the worker and each ingest child all call it."""
        with bare_root():
            configure_log_output("INFO")
            configure_log_output("DEBUG")
            (handler,) = logging.getLogger().handlers
            level = logging.getLogger("app").level

        assert sum(isinstance(f, PiiRedactionFilter) for f in handler.filters) == 1
        assert level == logging.DEBUG, "the second call sets the level again"

    def test_a_process_with_a_handler_of_its_own_is_left_alone(self):
        """An operator's `--log-config`, or a test runner's capture: as `basicConfig`
        does, a root handler somebody else installed means logging is theirs."""
        own = logging.StreamHandler(io.StringIO())

        with bare_root():
            logging.getLogger().addHandler(own)
            configure_log_output("DEBUG")
            handlers = list(logging.getLogger().handlers)
            level = logging.getLogger("app").level

        assert handlers == [own]
        assert level == logging.NOTSET


@pytest.mark.anyio
async def test_the_lifespan_prints_the_applications_log_at_the_configured_level(monkeypatch):
    """The API is the process whose log `docker compose logs app` shows."""
    from app import main
    from tests.test_lifespan_drain import _stub_startup

    _stub_startup(monkeypatch)
    monkeypatch.setattr(main.settings, "LOG_LEVEL", "DEBUG")

    async with main.lifespan(FastAPI()):
        pass

    main.configure_log_output.assert_called_once_with("DEBUG")
