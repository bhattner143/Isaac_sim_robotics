"""
project_utils/log_isaacsim.py
-------------------------------
Suppress Isaac Sim verbose startup output (extension loading, GPU tables,
carb warnings) so only Python-like errors and user prints are visible.

Usage::

    from project_utils.log_isaacsim import IsaacSimLogger

    log = IsaacSimLogger.from_argv()  # reads --verbose from CLI
    log.suppress()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    from omni.isaac.core import World  # keep suppressed during omni imports

    log.restore()   # normal output resumes here

    # ... user code prints normally ...

    log.close(app)  # suppress shutdown noise + close app

With ``--verbose`` on the command line the logger is a no-op and Isaac
Sim prints its full original output.
"""

import sys
import os
import atexit
import time as _time


class IsaacSimLogger:
    """Suppress Isaac Sim C++/Python noise via OS-level fd redirection.

    Isaac Sim's kit framework writes directly to file descriptors 1 and 2
    (stdout/stderr) from C++, bypassing Python's ``sys.stdout``.  Simple
    ``sys.stdout = ...`` replacement therefore does **not** mute the
    extension-loading, GPU-table, and carb-warning output.

    This class uses ``os.dup2()`` to redirect the underlying file
    descriptors to ``/dev/null`` during startup, then restores them.

    * **Default** (quiet): only user ``print()`` and Python tracebacks
      reach the terminal.
    * **--verbose**: original Isaac Sim CLI output, no suppression.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_argv(cls) -> "IsaacSimLogger":
        """Create a logger by reading ``--verbose`` from *sys.argv*."""
        verbose = "--verbose" in sys.argv
        return cls(verbose=verbose)

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._active = False
        self._start_time: float = 0.0
        self._saved_stdout_fd: int = -1
        self._saved_stderr_fd: int = -1
        self._devnull_fd: int = -1
        self._original_excepthook = sys.excepthook

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suppress(self) -> None:
        """Begin suppressing output.  Call **before** ``SimulationApp()``."""
        if self.verbose:
            return

        # Force carb to only log errors
        os.environ["CARB_LOG_LEVEL"] = "error"

        # Suppress Python-level DeprecationWarning (omni wrappers)
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Flush Python buffers before redirecting underlying fds
        sys.stdout.flush()
        sys.stderr.flush()

        # Save original file descriptors (dup returns a new fd)
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)

        # Redirect fd 1 and fd 2 to /dev/null
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull_fd, 1)
        os.dup2(self._devnull_fd, 2)

        self._active = True
        self._start_time = _time.monotonic()

        # If the process crashes, restore fds so the traceback is visible
        _self_ref = self

        def _on_crash(exc_type, exc_value, exc_tb):
            _self_ref._restore_fds()
            sys.stderr.write(
                "\n[IsaacSimLogger] Startup failed "
                "\u2014 re-run with --verbose for full output.\n\n"
            )
            sys.__excepthook__(exc_type, exc_value, exc_tb)

        sys.excepthook = _on_crash
        atexit.register(self._atexit_guard)

    def restore(self) -> None:
        """Restore normal output.  Call **after** all ``omni.*`` imports."""
        if not self._active:
            return

        elapsed = _time.monotonic() - self._start_time
        self._restore_fds()

        # Push carb / omni.log to error-only for remaining runtime
        try:
            import carb.settings  # available after SimulationApp
            settings = carb.settings.get_settings()
            settings.set("/log/outputStreamLevel", "error")
            settings.set("/log/level", "error")
        except Exception:
            pass
        try:
            import omni.log as _omni_log
            _omni_log.set_channel_enabled("omni.fabric.plugin", False, omni.log.SettingBehavior.OVERRIDE)
        except Exception:
            pass

        sys.stdout.write(
            f"Isaac Sim ready ({elapsed:.1f}s)"
            " \u2014 use --verbose for full startup log\n"
        )
        sys.stdout.flush()

    def close(self, app) -> None:
        """Suppress shutdown noise and close *SimulationApp*."""
        if not self.verbose:
            sys.stdout.flush()
            sys.stderr.flush()
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            saved_out = os.dup(1)
            saved_err = os.dup(2)
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)

        app.close()

        if not self.verbose:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
            os.close(devnull_fd)

    # ------------------------------------------------------------------
    # Escape hatch — print *through* suppression
    # ------------------------------------------------------------------

    def print(self, *args, **kwargs) -> None:
        """Print to real stdout even while suppressed."""
        if self._active and self._saved_stdout_fd >= 0:
            msg = " ".join(str(a) for a in args) + kwargs.get("end", "\n")
            os.write(self._saved_stdout_fd, msg.encode())
        else:
            print(*args, **kwargs)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def get_startup_log(self) -> str:
        """Return startup log path hint (use --verbose to see full output)."""
        return "Run with --verbose to see full Isaac Sim startup output."

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _restore_fds(self) -> None:
        """Restore original file descriptors."""
        if not self._active:
            return
        self._active = False

        # Restore original fds
        os.dup2(self._saved_stdout_fd, 1)
        os.dup2(self._saved_stderr_fd, 2)

        # Close saved copies and devnull
        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)
        if self._devnull_fd >= 0:
            os.close(self._devnull_fd)

        self._saved_stdout_fd = -1
        self._saved_stderr_fd = -1
        self._devnull_fd = -1

        sys.excepthook = self._original_excepthook

    def _atexit_guard(self) -> None:
        """Safety net: restore fds if *restore()* was never called."""
        if self._active:
            self._restore_fds()
