"""Ending scripts that opened MuJoCo's passive viewer."""
import os
import sys
import traceback
from typing import Callable


def exit_after(main: Callable[[], object]) -> None:
    """Run a script's main(), then end the process without Python's interpreter teardown.

    On some Linux desktops (seen on Wayland) MuJoCo's passive viewer segfaults or hangs in
    that teardown, even after its window closed cleanly. By the time main() returns,
    everything is written (episode logs are closed by their recorder), so nothing is lost by
    skipping it. The exit code is kept: 1 after an uncaught error, 130 after Ctrl-C.
    """
    code = 0
    try:
        main()
    except SystemExit as stop:
        if isinstance(stop.code, int):
            code = stop.code
        elif stop.code is not None:
            print(stop.code, file=sys.stderr)
            code = 1
    except KeyboardInterrupt:
        code = 130
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
