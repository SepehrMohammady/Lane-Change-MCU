#!/bin/bash
# Start the CARLA server without rendering, run one client script, stop the server (2026-11-11).
# WSL has no GPU Vulkan, so the server runs with -nullrhi: physics and traffic only, no cameras.
#   bash unas/run_carla.sh unas/carla_record.py --seed 1
# CARLA 0.9.16 (Linux package) is expected in ~/carla/0.9.16 and the client venv in ~/carla/env
# (python3 -m venv ~/carla/env; pip install numpy and the cp310 wheel from PythonAPI/carla/dist).
CARLA=${CARLA_ROOT:-$HOME/carla/0.9.16}
PY=${CARLA_PY:-$HOME/carla/env/bin/python}
cd "$(dirname "$0")/.."
pkill -f CarlaUE4-Linux-Shipping 2>/dev/null; sleep 1
"$CARLA/CarlaUE4.sh" -nullrhi -nosound -carla-rpc-port=2000 > "$HOME/carla/server.log" 2>&1 &
SP=$!
# the client holds the Traffic Manager; a client that hangs at exit must not block the caller
timeout -k 10 ${CARLA_TIMEOUT:-1800} "$PY" "$@"
RC=$?
# a server left in synchronous mode by a dead client does not react to SIGTERM
pkill -9 -f CarlaUE4-Linux-Shipping; wait $SP 2>/dev/null
exit $RC
