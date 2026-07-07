"""DMIR µNAS project — source package.

Modules:
    config      -- single experiment configuration dataclass (user-facing)
    data        -- pickle loading, sanitation, torch dataloaders
    models      -- baseline and NAS-produced architectures
    train       -- training / evaluation loops
    eda         -- exploratory data analysis plots
    log_utils   -- JSONL experiment logging + LOGBOOK entries
    env_utils   -- seeding, device selection, environment report
"""
