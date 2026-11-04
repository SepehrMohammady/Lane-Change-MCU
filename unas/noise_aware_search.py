"""Noise-aware variants of the fork's aging evolution (Part A of the Q1 plan, 2026-11-02).

One training run does not rank our candidates: on highD, the run the search kept and the
five-seed mean correlate at Spearman 0.17 (paper, "Single Runs Do Not Rank the Candidates").
This class keeps the fork's search space, mutations, scalarised fitness and resource model and
changes only how training runs are spent. The budget counts training runs, so every variant
gets the same compute.

  random    every candidate drawn at random, one run each; the final choice is made afterwards
            by unas/select_by_seeds.py, as for the fork's own search (150 runs + 60).
  fixed     every candidate trained NAS_FIXED_K times (default 3) and scored by its mean, as in
            Dushatskiy et al. (GECCO 2022); the choice is the feasible candidate with the best mean.
  adaptive  every new candidate gets one run. Extra runs go to feasible candidates that could still
            be the best: those whose lower bound mean - z*sigma/sqrt(n) reaches the incumbent's upper
            bound mean + z*sigma/sqrt(n), up to NAS_NMAX runs (default 5), the incumbent first and
            then by lower bound. After the initial population at most every second run is such a
            re-run; in the last NAS_FINAL_SHARE of the budget (default 0.15) every run is, while any
            is left. sigma is the pooled spread over seeds of the feasible candidates with two or more
            runs (NAS_SIGMA0, default 0.005, until five degrees of freedom exist). The choice is the
            feasible candidate with the best mean among those with NAS_NMAX runs.

"Feasible" means within the resource bounds of the config (peak memory, model size, MACs); the
error bound only shapes the fitness, as in the fork. Only validation error is read; the fork's
test error is never used.

Environment: NAS_MODE (random|fixed|adaptive), NAS_BUDGET (training runs), NAS_FIXED_K, NAS_NMAX,
NAS_Z, NAS_SIGMA0, NAS_FINAL_SHARE, NAS_LOG (JSONL file, one line per training run).

Writes to the checkpoint folder: <name>_noiseaware_state.pickle (resumable between chunks) and,
for fixed and adaptive, <name>_final.json and <name>_final.h5 (the chosen network, untrained,
for unas/final_eval.py).
"""
from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import ray

from uNAS.search_algorithms.aging_evolution import AgingEvoSearch, EvaluatedPoint
from uNAS.utils import Scheduler, debug_mode

MODES = ("random", "fixed", "adaptive")
FAILED = 10 ** 12          # resource value the patched fork returns for a candidate that failed


@dataclass
class Candidate:
    cid: int
    point: object                                   # the fork's ArchitecturePoint
    runs: list = field(default_factory=list)        # validation errors of the completed runs
    resources: list | None = None                   # [peak memory, model size, MACs]
    scheduled: int = 0                              # runs submitted, finished or not
    failed: bool = False
    in_pop: bool = False

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def mean(self) -> float:
        return float(np.mean(self.runs)) if self.runs else math.inf


class NoiseAwareSearch(AgingEvoSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        env = os.environ
        self.mode = env.get("NAS_MODE", "adaptive")
        if self.mode not in MODES:
            raise ValueError(f"NAS_MODE must be one of {MODES}, not {self.mode!r}")
        self.budget = int(env.get("NAS_BUDGET", "210"))
        self.fixed_k = int(env.get("NAS_FIXED_K", "3"))
        self.n_max = int(env.get("NAS_NMAX", "5"))
        self.z = float(env.get("NAS_Z", "1.0"))
        self.sigma0 = float(env.get("NAS_SIGMA0", "0.005"))
        self.final_share = float(env.get("NAS_FINAL_SHARE", "0.15"))
        self.log_path = env.get("NAS_LOG")
        self.cands: dict[int, Candidate] = {}
        self.pop: list[int] = []        # aging population, candidate ids, oldest first
        self.done = 0                   # finished training runs (the budget counts these)
        self.n_new = 0                  # new candidates submitted after the initial population
        self.n_re = 0                   # re-runs submitted
        self.state_file = self.root_dir / f"{self.experiment_name}_noiseaware_state.pickle"

    # ------------------------------------------------------------------ state
    def _save(self):
        tmp = self.state_file.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"mode": self.mode, "budget": self.budget, "cands": self.cands, "pop": self.pop,
                         "done": self.done, "n_new": self.n_new, "n_re": self.n_re,
                         "rng": np.random.get_state()}, f)
        os.replace(tmp, self.state_file)

    def _load(self):
        with open(self.state_file, "rb") as f:
            s = pickle.load(f)
        if (s["mode"], s["budget"]) != (self.mode, self.budget):
            raise RuntimeError(f"{self.state_file} belongs to {s['mode']} with budget {s['budget']}")
        # runs in flight when the previous chunk ended are lost; so are candidates that never finished one
        self.cands = {i: c for i, c in s["cands"].items() if c.n}
        for c in self.cands.values():
            c.scheduled = c.n
        self.pop = [i for i in s["pop"] if i in self.cands]
        self.done, self.n_new, self.n_re = s["done"], s["n_new"], s["n_re"]
        np.random.set_state(s["rng"])
        self.log.info(f"Resumed {self.mode}: {self.done} of {self.budget} runs, {len(self.cands)} candidates.")

    # ------------------------------------------------------------------ statistics
    def _feasible(self, c: Candidate) -> bool:
        if c.failed or c.resources is None or not c.runs:
            return False
        return all(r <= b for r, b in zip(c.resources, self.constraint_bounds[1:]) if b is not None)

    def _sigma(self) -> float:
        ss = dof = 0.0
        for c in self.cands.values():
            if c.n >= 2 and self._feasible(c):
                ss += float(np.var(c.runs, ddof=1)) * (c.n - 1)
                dof += c.n - 1
        return math.sqrt(ss / dof) if dof >= 5 else self.sigma0

    def _bound(self, c: Candidate, sigma: float, sign: int) -> float:
        return c.mean + sign * self.z * sigma / math.sqrt(c.n)

    def _unresolved(self) -> list[Candidate]:
        feas = [c for c in self.cands.values() if self._feasible(c)]
        if not feas:
            return []
        sigma = self._sigma()
        inc = min(feas, key=lambda c: c.mean)
        top = self._bound(inc, sigma, +1)
        out = [c for c in feas if c.scheduled < self.n_max and (c is inc or self._bound(c, sigma, -1) <= top)]
        out.sort(key=lambda c: (c is not inc, self._bound(c, sigma, -1)))
        return out

    # ------------------------------------------------------------------ choosing the next run
    def _as_evaluated(self, c: Candidate) -> EvaluatedPoint:
        return EvaluatedPoint(point=c.point, val_error=c.mean, test_error=math.nan, resource_features=c.resources)

    def _new_candidate(self) -> Candidate:
        if self.mode == "random" or len(self.cands) < self.initial_population_size:
            point = self.random_sample()
        else:
            sample = np.random.choice(self.pop, size=self.sample_size)
            parent = max((self._as_evaluated(self.cands[int(i)]) for i in sample), key=self.get_mo_fitness_fn())
            point = self.evolve(parent.point)
        cid = max(self.cands, default=-1) + 1
        c = self.cands[cid] = Candidate(cid=cid, point=point)
        return c

    def _next_job(self, submitted: int):
        """(candidate, kind) for the next training run, or None to wait for a result."""
        initial_done = len(self.pop) >= self.initial_population_size
        if self.mode == "fixed":
            for c in self.cands.values():
                if c.scheduled < self.fixed_k and not c.failed:
                    return c, "repeat"
        if self.mode == "adaptive" and initial_done:
            final = self.budget - submitted <= self.final_share * self.budget
            if final or self.n_re < self.n_new:
                todo = self._unresolved()
                if todo:
                    return todo[0], "re-run"
        if len(self.cands) >= self.initial_population_size and not initial_done:
            return None                     # the fork also waits for the whole initial population
        c = self._new_candidate()
        if initial_done:
            self.n_new += 1
        return c, "new"

    # ------------------------------------------------------------------ results
    def _record(self, info: EvaluatedPoint):
        c = self.cands.get(getattr(info.point, "cid", None))
        if c is None:                       # a run from before a restart whose candidate was dropped
            return
        self.done += 1
        kind = getattr(info.point, "kind", "?")
        failed = info.resource_features[0] >= FAILED
        resources = [x.item() if hasattr(x, "item") else x for x in info.resource_features]
        if failed:
            if not c.runs:                  # scored as the worst, as the patched fork does
                c.failed, c.runs, c.resources = True, [1.0], resources
        else:
            if c.resources is None:
                c.resources = resources
            c.runs.append(float(info.val_error))
        joins = c.failed or (c.n == self.fixed_k if self.mode == "fixed" else c.n == 1)
        if joins and not c.in_pop:
            c.in_pop = True
            self.pop.append(c.cid)
            while len(self.pop) > self.population_size:
                self.pop.pop(0)
        if self.log_path:
            rec = {"search": self.experiment_name, "mode": self.mode, "run": self.done, "candidate": c.cid,
                   "kind": kind, "val_error": None if failed else float(info.val_error), "n": c.n,
                   "mean": None if c.failed else c.mean, "feasible": self._feasible(c),
                   "resources": c.resources, "sigma": self._sigma(),
                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------ search
    def search(self, load_from: str = None, save_every: int = None):
        if self.state_file.exists():
            self._load()
        ray.init(local_mode=debug_mode())
        trainer = ray.put(self.trainer)
        ss = ray.put(self.config.search_space)
        scheduler = Scheduler([self.gpu_trainer.remote(ss, trainer, self.model_saver)
                               for _ in range(self.max_parallel_evaluations)])
        self.log.info(f"{self.mode} search, {self.budget} training runs, {self.max_parallel_evaluations} workers.")

        while self.done < self.budget:
            submitted = self.done + scheduler.pending_tasks()
            job = None
            if scheduler.has_a_free_worker() and submitted < self.budget:
                job = self._next_job(submitted)
            if job is not None:
                c, kind = job
                c.scheduled += 1
                c.point.cid, c.point.kind = c.cid, kind
                if kind == "re-run":
                    self.n_re += 1
                self.log.info(f"Run {submitted + 1}/{self.budget}: candidate {c.cid}, {kind} (run {c.scheduled}).")
                scheduler.submit(c.point)
            elif scheduler.pending_tasks():
                self._record(scheduler.await_any())
                self._save()
            else:
                break
        self._save()
        if self.mode != "random":
            self._finish()
        print("Search done!")

    def _finish(self):
        need = self.fixed_k if self.mode == "fixed" else self.n_max
        feas = [c for c in self.cands.values() if self._feasible(c)]
        if not feas:
            self.log.error("No feasible candidate; nothing chosen.")
            return
        most = max(c.n for c in feas)
        pool = [c for c in feas if c.n >= min(need, most)]
        best = min(pool, key=lambda c: c.mean)
        data = self.trainer.dataset() if callable(self.trainer.dataset) else self.trainer.dataset
        model = self.config.search_space.to_keras_model(best.point.arch, data.input_shape, data.num_classes)
        h5 = self.root_dir / f"{self.experiment_name}_final.h5"
        model.save(h5.as_posix())
        ranked = sorted(feas, key=lambda c: c.mean)
        runs = [c.n for c in self.cands.values() if not c.failed]
        out = {
            "search": self.experiment_name, "mode": self.mode, "budget": self.budget, "runs_done": self.done,
            "candidates": len(self.cands), "failed": sum(c.failed for c in self.cands.values()),
            "feasible": len(feas), "re_runs": self.n_re,
            "runs_per_candidate": {str(k): runs.count(k) for k in sorted(set(runs))},
            "sigma": self._sigma(), "z": self.z, "n_max": self.n_max, "fixed_k": self.fixed_k,
            "final_share": self.final_share, "rule": f"lowest mean validation error among feasible candidates "
                                                    f"with {min(need, most)} or more runs",
            "chosen": {"candidate": best.cid, "runs": best.runs, "mean": best.mean, "resources": best.resources,
                       "params": int(model.count_params()), "h5": h5.name},
            "top": [{"candidate": c.cid, "n": c.n, "mean": c.mean, "resources": c.resources} for c in ranked[:15]],
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (self.root_dir / f"{self.experiment_name}_final.json").write_text(json.dumps(out, indent=1, default=str))
        self.log.info(f"Chosen: candidate {best.cid}, {best.n} runs, mean validation error {best.mean:.4f}, "
                      f"{model.count_params()} parameters.")
