"""Control native (C-level) stdout/stderr produced by rendering backends.

Libraries such as Blender write progress and diagnostics directly to file
descriptors 1/2 from C code, bypassing Python's ``sys.stdout`` / ``sys.stderr``
and the :mod:`logging` module. ``contextlib.redirect_stdout`` therefore cannot
capture them — only OS-level file-descriptor redirection can.

:func:`manage_native_output` redirects those descriptors for the duration of a
block and, based on the supplied logger's level, either discards the output or
re-emits it line by line through the logger.
"""

import contextlib
import logging
import os
import sys
import threading


@contextlib.contextmanager
def manage_native_output(target_logger: logging.Logger, *, prefix: str = ""):
    """Suppress or route C-level ``stdout``/``stderr`` (fd 1/2) for a block.

    Behaviour is keyed off ``target_logger`` so callers do not need a separate
    switch:

    - When ``target_logger`` is enabled for :data:`logging.DEBUG`, each captured
      line is re-emitted via ``target_logger.debug`` (a dedicated ``"<name>.native"``
      child logger, optionally tagged with ``prefix``).
    - Otherwise the native output is discarded (sent to ``os.devnull``).

    File descriptors 1 and 2 are always restored on exit.

    Args:
        target_logger: Logger whose effective level decides route-vs-suppress and
            whose name roots the child logger used for routed lines.
        prefix: Optional tag prepended to each routed line (e.g. ``"blender"``).
    """
    # Flush Python-level buffers before swapping the underlying descriptors.
    sys.stdout.flush()
    sys.stderr.flush()

    saved_out = os.dup(1)
    saved_err = os.dup(2)

    def _restore():
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)

    if not target_logger.isEnabledFor(logging.DEBUG):
        # Suppress: point fd 1/2 at the null device.
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
        finally:
            os.close(devnull)
        try:
            yield
        finally:
            _restore()
        return

    # Route: capture fd 1/2 through a pipe and re-emit via the logger. The sink
    # writes to a *copy of the original* stderr so the logger's own output never
    # feeds back into the pipe we just installed (which would loop forever).
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    fmt = f"[{prefix}] %(message)s" if prefix else "%(message)s"
    sink = logging.StreamHandler(os.fdopen(os.dup(saved_err), "w", closefd=True))
    sink.setFormatter(logging.Formatter(fmt))
    native_logger = logging.getLogger(f"{target_logger.name}.native")
    prev = (native_logger.handlers, native_logger.propagate, native_logger.level)
    native_logger.handlers = [sink]
    native_logger.propagate = False
    native_logger.setLevel(logging.DEBUG)

    def pump():
        with os.fdopen(read_fd, "r", errors="replace") as reader:
            for line in reader:
                line = line.rstrip("\n")
                if line:
                    native_logger.debug(line)

    thread = threading.Thread(target=pump, name="native-output", daemon=True)
    thread.start()
    try:
        yield
    finally:
        # Restoring fd 1/2 drops the last references to the pipe's write end,
        # so the reader sees EOF and the pump thread finishes.
        _restore()
        thread.join(timeout=5)
        sink.close()
        native_logger.handlers, native_logger.propagate, native_logger.level = prev
