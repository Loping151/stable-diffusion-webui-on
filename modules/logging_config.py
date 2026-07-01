import logging
import os
import sys
import time

try:
    from tqdm import tqdm


    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, fallback_handler: logging.Handler):
            super().__init__()
            self.fallback_handler = fallback_handler

        def emit(self, record):
            try:
                # If there are active tqdm progress bars,
                # attempt to not interfere with them.
                if tqdm._instances:
                    tqdm.write(self.format(record))
                else:
                    self.fallback_handler.emit(record)
            except Exception:
                self.fallback_handler.emit(record)

except ImportError:
    TqdmLoggingHandler = None


class _TimestampedStream:
    """Wrap a text stream so each fresh line is prefixed with a wall-clock timestamp.

    Most of Forge's console output comes from bare ``print()`` calls, which the logging
    formatter never sees; wrapping stdout is what actually puts a time on those lines. Writes
    that contain a carriage return (``\\r`` — tqdm / progress-bar redraws) pass through untouched
    so progress bars still render in place instead of being spammed with stamps.
    """

    def __init__(self, stream, fmt="[%H:%M:%S] "):
        self._stream = stream
        self._fmt = fmt
        self._at_line_start = True

    def write(self, text):
        if not text:
            return 0
        # Progress redraws (tqdm etc.) use '\r'; never inject a stamp into those.
        if "\r" in text:
            self._at_line_start = text.endswith("\n")
            return self._stream.write(text)
        stamp = time.strftime(self._fmt)
        out = []
        for line in text.splitlines(keepends=True):
            if self._at_line_start and line.strip("\n"):
                out.append(stamp)
            out.append(line)
            self._at_line_start = line.endswith("\n")
        return self._stream.write("".join(out))

    def flush(self):
        self._stream.flush()

    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        # Delegate everything else (encoding, buffer, fileno, reconfigure, ...) to the real stream.
        return getattr(self._stream, name)


def _install_stdout_timestamps():
    """Prefix print()/stdout lines with a timestamp. Idempotent; opt out via SD_WEBUI_NO_TIMESTAMP."""
    if os.environ.get("SD_WEBUI_NO_TIMESTAMP"):
        return
    if isinstance(sys.stdout, _TimestampedStream):
        return
    try:
        sys.stdout = _TimestampedStream(sys.stdout)
    except Exception:
        # Never let logging setup take down the process.
        pass


def setup_logging(loglevel):
    if loglevel is None:
        loglevel = os.environ.get("SD_WEBUI_LOG_LEVEL")

    # Timestamp bare print()/stdout output regardless of --loglevel: that is where the bulk of
    # Forge's console log lives, so this is what makes the log readable over time.
    _install_stdout_timestamps()

    if logging.root.handlers:
        # Already configured, do not interfere
        return

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s [%(name)s] %(message)s',
        '%Y-%m-%d %H:%M:%S',
    )

    if os.environ.get("SD_WEBUI_RICH_LOG"):
        from rich.logging import RichHandler
        handler = RichHandler()
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

    if TqdmLoggingHandler:
        handler = TqdmLoggingHandler(handler)

    handler.setFormatter(formatter)

    # Always attach the timestamped handler so logging records get a consistent, dated format.
    # When no level was requested, default to WARNING (the effective visibility Forge had before,
    # via logging's lastResort) so we add timestamps/structure without unleashing third-party
    # INFO/DEBUG noise; pass --loglevel INFO/DEBUG to see more.
    log_level = getattr(logging, loglevel.upper(), None) if loglevel else logging.WARNING
    if log_level is None:
        log_level = logging.WARNING
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)
