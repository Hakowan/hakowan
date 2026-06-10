"""Tests for hakowan.common.output.manage_native_output.

The context manager redirects OS file descriptors 1/2, which conflicts with
pytest's own fd capture, so each scenario is exercised in a clean subprocess.
"""

import subprocess
import sys
import textwrap


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )


class TestManageNativeOutput:
    def test_suppress_when_not_debug(self):
        # Logger at INFO -> native (fd-level) writes are discarded.
        result = _run(
            """
            import logging, os, sys
            from hakowan.common.output import manage_native_output
            log = logging.getLogger("hakowan")
            log.setLevel(logging.INFO)
            with manage_native_output(log, prefix="blender"):
                os.write(1, b"SUPPRESSED_STDOUT\\n")
                os.write(2, b"SUPPRESSED_STDERR\\n")
            sys.stderr.write("AFTER_BLOCK\\n"); sys.stderr.flush()
            """
        )
        combined = result.stdout + result.stderr
        assert "AFTER_BLOCK" in result.stderr  # fds restored afterwards
        assert "SUPPRESSED_STDOUT" not in combined
        assert "SUPPRESSED_STDERR" not in combined

    def test_route_when_debug(self):
        # Logger at DEBUG -> native writes are re-emitted through the logger.
        result = _run(
            """
            import logging, os, sys
            from hakowan.common.output import manage_native_output
            log = logging.getLogger("hakowan")
            log.setLevel(logging.DEBUG)
            with manage_native_output(log, prefix="blender"):
                os.write(1, b"ROUTED_STDOUT\\n")
                os.write(2, b"ROUTED_STDERR\\n")
            sys.stderr.write("AFTER_BLOCK\\n"); sys.stderr.flush()
            """
        )
        assert "AFTER_BLOCK" in result.stderr
        # Routed lines are prefixed and land on the original stderr.
        assert "[blender] ROUTED_STDOUT" in result.stderr
        assert "[blender] ROUTED_STDERR" in result.stderr

    def test_fds_restored_after_exception(self):
        # An exception inside the block must still restore fd 1/2.
        result = _run(
            """
            import logging, os, sys
            from hakowan.common.output import manage_native_output
            log = logging.getLogger("hakowan")
            log.setLevel(logging.INFO)
            try:
                with manage_native_output(log, prefix="blender"):
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
            print("STDOUT_WORKS")
            sys.stderr.write("STDERR_WORKS\\n"); sys.stderr.flush()
            """
        )
        assert "STDOUT_WORKS" in result.stdout
        assert "STDERR_WORKS" in result.stderr
        assert result.returncode == 0
