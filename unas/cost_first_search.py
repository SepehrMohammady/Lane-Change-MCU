"""Noise-aware cost-first parent selection for the fork's aging evolution (2026-11-10).

The corrected searches reach an accuracy plateau: the two search seeds of every task agree in accuracy
while their chosen networks differ 1.8-4.5x in MACs, and one training run cannot rank candidates inside
the plateau (Spearman 0.10-0.55 between single-run and five-seed validation over the shortlists). The
fork's fitness divides each objective by its bound and weights the objectives at random, so while no
candidate reaches the error bound the error term decides most parent choices and the population drifts
towards the cost budget.

This class changes only the parent selection. A candidate is on the plateau if it is within the resource
bounds and its validation error is at most e* + CF_TAU, where e* is the lowest validation error among the
evaluated candidates within the bounds. On the plateau, fewer MACs is fitter, and every plateau candidate
beats every other candidate; the others keep the fork's scalarised fitness. CF_TAU (in validation error,
1 - accuracy) is twice the single-run standard deviation measured on earlier five-seed retrains. Space,
mutations, resource model, budgets, training and the number of training runs are the fork's; the state
file is the fork's, so chunked runs resume as for the fork's own search.

Used by the *_v4cf configs (unas/highd_config.py, unas/dmir_config.py) through unas/run_noise_aware.sh,
mode costfirst; the final choice is made by unas/select_by_seeds.py with SEL_RULE=costfirst.
"""
import os

from uNAS.search_algorithms.aging_evolution import AgingEvoSearch


class CostFirstSearch(AgingEvoSearch):
    def _within_bounds(self, point) -> bool:
        # resource_features: peak memory, model size, MACs; constraint_bounds: error, then the same three
        return all(b is None or f <= b for f, b in zip(point.resource_features, self.constraint_bounds[1:]))

    def get_mo_fitness_fn(self):
        tau = float(os.environ["CF_TAU"])
        base = super().get_mo_fitness_fn()          # the fork's fitness, with fresh random weights
        feasible = [p.val_error for p in self.history if self._within_bounds(p)]
        ref = min(feasible) if feasible else None
        mac_bound = self.constraint_bounds[3]

        def fitness(p):
            if ref is not None and self._within_bounds(p) and p.val_error <= ref + tau:
                return -p.resource_features[2] / mac_bound      # in [-1, 0): fewer MACs is fitter
            return -1.0 + base(p)                               # base(p) < 0: below every plateau candidate
        return fitness
