#!/bin/bash
# Guard study (unas/online_guard.py), five seeds, two processes at a time (memory). Run in WSL from the repo.
cd /mnt/c/Projects/PhD/DIMIR
run() { SEEDS=$1 OUT_NAME=online_guard_s$1.jsonl ~/dmir_nas/bin/python unas/online_guard.py run > /tmp/guard_s$1.log 2>&1; }
run 100 & run 101 & wait
run 102 & run 103 & wait
run 104
echo done
