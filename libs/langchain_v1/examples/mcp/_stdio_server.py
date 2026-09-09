"""Run one of the shared servers over stdio, chosen by argv.

`MCPAdapter` launches this as a subprocess; it is not part of the API being
demonstrated.
"""

import sys

from _servers import run_calculator_stdio, run_weather_stdio

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "weather"
    (run_calculator_stdio if which == "calculator" else run_weather_stdio)()
