"""Full paper RCR: counterfactual four-dimensional BCD and capacity CLC.

Defaults match the manuscript: tau=0.1, lambda=0.3, B=8, G_max=8.
HGNN's reward is in raw time units, hence sigma=1. Configure ablations
through RCR_CONFIG (JSON), or matching env/model parameters under "rcr".
The original Memory.states field stores factual snapshots, NEVER BCD.
All static instance data, feasible horizons and constraint sets are frozen.
BCD: J/G/W weighted pressure changes and tau * KL(parent || successor),
with fixed parent support, balanced family weights and action-wise z-score.
CLC: H * relu(max pressure over ALL valid constraints - epsilon_c).
Time bins define ALL endpoint-pair intervals, not only adjacent bins.
The encoder, critic, native feature normalization and PPO objective are intact.
"""
from dataclasses import dataclass
import json
import os

import numpy as np
import torch


@dataclass(frozen=True)
class RCRConfig:
    enabled: bool = True
    actor_features: bool = True
    reward_shaping: bool = True
    extra_group_count: int = 8
    window_bins: int = 8
    pressure_temperature: float = 0.1
    reward_weight: float = 0.3
    reward_scale: float = 1.0
    candidate_chunk_size: int = 32
    certificate_epsilon: float = 1e-7
    actor_feature_floor: float = 1e-6
    actor_feature_clip: float = 5.0

    def __post_init__(self):
        for key in ('enabled', 'actor_features', 'reward_shaping'):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(key + ' must be bool')
        for key, minimum in (('extra_group_count', 0), ('window_bins', 1),
                             ('candidate_chunk_size', 1)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(key + ' has an invalid integer value')
        for key in ('pressure_temperature', 'reward_scale', 'actor_feature_floor',
                    'actor_feature_clip', 'reward_weight', 'certificate_epsilon'):
            value = getattr(self, key)
            minimum_ok = value >= 0 if key in ('reward_weight', 'certificate_epsilon') else value > 0
            if not np.isfinite(value) or not minimum_ok:
                raise ValueError(key + ' has an invalid value')

    @property
    def use_actor(self):
        return self.enabled and self.actor_features

    @property
    def use_reward(self):
        return self.enabled and self.reward_shaping


def get_rcr_config(options=None):
    if isinstance(options, RCRConfig):
        return options
    if options is None:
        options = json.loads(os.environ.get('RCR_CONFIG', '{}'))
    options = dict(options or {})
    for short, full in (('tau', 'pressure_temperature'), ('lambda', 'reward_weight'),
                        ('B', 'window_bins'), ('G_max', 'extra_group_count'),
                        ('epsilon_z', 'actor_feature_floor'), ('B_z', 'actor_feature_clip'),
                        ('epsilon_c', 'certificate_epsilon'), ('sigma', 'reward_scale')):
        if short in options:
            if full in options:
                raise ValueError('Specify only one of ' + short + ' and ' + full)
            options[full] = options.pop(short)
    return RCRConfig(**options)


def _numpy(tensor):
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)


class SchedulingAdapter:
    """Read HGNN's authoritative event-clock state; never alter its simulator."""

    def __init__(self, env):
        self.duration = _numpy(env.old_proc_times_batch).astype(np.float64, copy=True)
        self.eligible = _numpy(env.old_ope_ma_adj_batch).astype(bool, copy=True)
        self.batch_size, self.n, self.m = self.duration.shape
        self.counts = _numpy(env.nums_opes).astype(np.int64)
        self.j = env.num_jobs
        self.first = _numpy(env.num_ope_biases_batch).astype(np.int64)
        self.last = _numpy(env.end_ope_biases_batch).astype(np.int64)
        self.op_valid = np.arange(self.n)[None, :] < self.counts[:, None]
        if (not np.isfinite(self.duration).all() or (self.duration < 0).any()
                or not (self.eligible & (self.duration > 0)).any(-1)[self.op_valid].all()
                or not np.array_equal(self.eligible[self.op_valid],
                                      (self.duration > 0)[self.op_valid])):
            raise ValueError('HGNN RCR requires positive processing times for eligible real operations.')
        self.pmin = np.where(self.op_valid,
                             np.where(self.eligible, self.duration, np.inf).min(-1), 0.)
        self.job_valid = _numpy(env.nums_ope_batch) > 0
        self.job_member = np.zeros((self.batch_size, self.j, self.n), dtype=np.float64)
        self.op_job = np.zeros((self.batch_size, self.n), dtype=np.int64)
        self.prefix = np.zeros((self.batch_size, self.n), dtype=np.float64)
        self.suffix = np.zeros_like(self.prefix)
        for b in range(self.batch_size):
            if self.counts[b] != self.job_valid[b].dot(_numpy(env.nums_ope_batch)[b]):
                raise ValueError('Invalid HGNN job lengths.')
            for j in range(self.j):
                if not self.job_valid[b, j]:
                    continue
                ops = np.arange(self.first[b, j], self.last[b, j] + 1)
                self.job_member[b, j, ops] = 1.
                self.op_job[b, ops] = j
                work = self.pmin[b, ops]
                self.prefix[b, ops] = np.cumsum(work) - work
                self.suffix[b, ops] = work.sum() - np.cumsum(work)
        self.total_work = np.einsum('bjn,bn->bj', self.job_member, self.pmin)
        # Two fixed, deterministic feasible witnesses (SPT/MWKR); no env/RNG mutation.
        self.horizon = np.minimum(self._witness('spt'), self._witness('mwkr'))
        if (self.horizon <= 0).any():
            raise ValueError('A feasible witness must have positive makespan.')

    def _witness(self, rule):
        horizons = np.zeros(self.batch_size, dtype=np.float64)
        for b in range(self.batch_size):
            next_op = self.first[b].copy()
            ready_j = np.zeros(self.j)
            ready_m = np.zeros(self.m)
            for _ in range(int(self.counts[b])):
                choices = []
                for j in range(self.j):
                    op = next_op[j]
                    if op > self.last[b, j]:
                        continue
                    for m in np.flatnonzero(self.eligible[b, op]):
                        p = self.duration[b, op, m]
                        finish = max(ready_j[j], ready_m[m]) + p
                        remaining_work = self.total_work[b, j] - self.prefix[b, op]
                        primary = p if rule == 'spt' else -remaining_work
                        choices.append((primary, finish, j, int(m), finish))
                _, _, j, m, finish = min(choices)
                ready_j[j] = ready_m[m] = finish
                next_op[j] += 1
            horizons[b] = ready_j.max()
        return horizons


@dataclass
class RCRState:
    """Sufficient factual state; static core is shared, dynamic tensors are owned."""
    core: object
    ids: torch.Tensor
    remaining: torch.Tensor
    job_ready: torch.Tensor
    machine_ready: torch.Tensor
    time: torch.Tensor

    def select(self, rows):
        return RCRState(self.core, self.ids[rows], self.remaining[rows],
                        self.job_ready[rows], self.machine_ready[rows], self.time[rows])

    def features(self, eligible, next_ops):
        return self.core.actor_features(self, eligible, next_ops)


class CapacityCore:
    """Fixed constraints on the environment's device; no policy or mutable cache."""

    def __init__(self, adapter, config, device):
        self.cfg = config
        a = adapter
        self.j, self.n, self.m = a.j, a.n, a.m
        self.device = device

        def tensor(value, dtype=torch.float64):
            return torch.as_tensor(value, dtype=dtype, device=device)

        self.duration = tensor(a.duration)
        self.eligible = tensor(a.eligible, torch.bool)
        self.pmin = tensor(a.pmin)
        self.op_valid = tensor(a.op_valid, torch.bool)
        self.job_valid = tensor(a.job_valid, torch.bool)
        self.job_member = tensor(a.job_member)
        self.op_job = tensor(a.op_job, torch.long)
        self.first = tensor(a.first, torch.long)
        self.last = tensor(a.last, torch.long)
        self.prefix = tensor(a.prefix)
        self.suffix = tensor(a.suffix)
        self.total_work = tensor(a.total_work)
        self.horizon = tensor(a.horizon)
        self.counts = tensor(a.counts, torch.long)

        # Original eligibility is immutable; deleting assigned arcs in HGNN
        # must never change these instance-level groups or minimum durations.
        groups = []
        for b in range(a.batch_size):
            base = [(m,) for m in range(a.m)]
            all_m = tuple(range(a.m))
            if all_m not in base:
                base.append(all_m)
            candidates = {tuple(np.flatnonzero(row)) for row in a.eligible[b, a.op_valid[b]]}
            candidates = [s for s in candidates if 1 < len(s) < a.m]

            def rank(group):
                member = np.zeros(a.m, dtype=bool)
                member[list(group)] = True
                restricted = (~a.eligible[b] | member).all(-1) & a.op_valid[b]
                return (-a.pmin[b, restricted].sum() / len(group), group)

            groups.append(base + sorted(candidates, key=rank)[:config.extra_group_count])
        self.groups = groups
        self.g = max(map(len, groups))
        machine_member = np.zeros((a.batch_size, self.g, a.m))
        group_valid = np.zeros((a.batch_size, self.g), dtype=bool)
        for b, sets in enumerate(groups):
            for g, machines in enumerate(sets):
                machine_member[b, g, list(machines)] = 1.
                group_valid[b, g] = True
        self.machine_member = tensor(machine_member)
        self.group_valid = tensor(group_valid, torch.bool)
        self.group_size = self.machine_member.sum(-1).clamp_min(1)
        self.op_member = ((~self.eligible[:, None] |
                           self.machine_member[:, :, None].bool()).all(-1)
                          & self.group_valid[:, :, None] & self.op_valid[:, None]).double()
        bins = config.window_bins
        windows = [(0., 1.)] + [(i / bins, j / bins) for i in range(bins)
                               for j in range(i + 1, bins + 1) if (i, j) != (0, bins)]
        self.windows = self.horizon[:, None, None] * tensor(windows)[None]
        self.w = len(windows)
        self.job_end, self.global_end = a.j, a.j + self.g
        self.valid = torch.cat((self.job_valid, self.group_valid,
                               self.group_valid.repeat_interleave(self.w - 1, dim=1)), dim=1)
        self.families = (slice(0, self.job_end), slice(self.job_end, self.global_end),
                         slice(self.global_end, self.valid.size(1)))

    @torch.no_grad()
    def snapshot(self, env):
        ids = torch.arange(self.counts.numel(), device=self.device)
        next_op = env.ope_step_batch
        prev = torch.minimum(torch.maximum(next_op - 1, self.first), self.last)
        end = env.schedules_batch[:, :, 3].double()
        ready_j = torch.where(next_op > self.first, end.gather(1, prev), 0.)
        return RCRState(self, ids, self.op_valid & (env.schedules_batch[:, :, 0] < .5),
                        ready_j, env.machines_batch[:, :, 1].detach().double().clone(),
                        env.time.detach().double().clone())

    @torch.no_grad()
    def constraints(self, state):
        """All constructed pressures and CLC, without BCD's active filtering."""
        ids = state.ids
        h = self.horizon[ids, None]
        pmin = self.pmin[ids]
        remaining = state.remaining.double()
        work = torch.bmm(self.job_member[ids], (remaining * pmin).unsqueeze(-1)).squeeze(-1)
        completed = self.total_work[ids] - work
        # The event clock rules out insertion into past idle intervals. Keep
        # completed job tails, including at terminal states, for sound CLC.
        ready = torch.where(work > 0, torch.maximum(state.job_ready, state.time[:, None]),
                            state.job_ready)
        earliest = (ready - completed).gather(1, self.op_job[ids]) + self.prefix[ids]
        latest = h - self.suffix[ids]
        left, right = self.windows[ids, :, 0], self.windows[ids, :, 1]
        mandatory = torch.minimum(pmin[:, :, None], (right - left)[:, None])
        mandatory = torch.minimum(mandatory, earliest[:, :, None] + pmin[:, :, None] - left[:, None])
        mandatory = torch.minimum(mandatory, right[:, None] - latest[:, :, None] + pmin[:, :, None])
        mandatory = mandatory.clamp_min(0) * remaining[:, :, None]
        demand = torch.bmm(self.op_member[ids], mandatory)
        machine_ready = torch.maximum(state.machine_ready, state.time[:, None])
        free = (right[:, None] - torch.maximum(left[:, None], machine_ready[:, :, None])).clamp_min(0)
        capacity = torch.bmm(self.machine_member[ids], free)
        violation = demand - capacity
        n_group = torch.bmm(self.op_member[ids], remaining.unsqueeze(-1)).squeeze(-1).clamp_min(1)
        x_job = (ready + work - h) / h
        x_global = violation[:, :, 0] / (h * self.group_size[ids])
        x_local = violation[:, :, 1:] / (h[:, :, None] * n_group[:, :, None])
        x = torch.cat((x_job, x_global, x_local.flatten(1)), dim=1)
        x = x.masked_fill(~self.valid[ids], 0.)
        gamma = x.masked_fill(~self.valid[ids], -torch.inf).max(-1).values
        delta = self.horizon[ids] * (gamma - self.cfg.certificate_epsilon).clamp_min(0)
        return x, delta

    def active_constraints(self, state, eligible):
        ids = state.ids
        remaining = state.remaining.double()
        live_jobs = torch.bmm(self.job_member[ids], remaining.unsqueeze(-1)).squeeze(-1) > 0
        live_groups = (torch.bmm(self.op_member[ids], remaining.unsqueeze(-1)).squeeze(-1) > 0)
        live_groups &= self.group_valid[ids]
        starts = torch.maximum(state.job_ready[:, :, None], state.machine_ready[:, None])
        starts = torch.maximum(starts, state.time[:, None, None])
        phi = starts.masked_fill(~eligible, torch.inf).flatten(1).min(-1).values
        future = self.windows[ids, 1:, 1] > phi[:, None]
        return self.valid[ids] & torch.cat((live_jobs, live_groups,
                          (live_groups[:, :, None] & future[:, None]).flatten(1)), dim=1)

    def log_weights(self, x, active):
        """Equal family mass; softmax only within each nonempty family."""
        present = torch.stack([active[:, part].any(-1) for part in self.families], dim=1)
        log_k = present.sum(-1).clamp_min(1).double().log()[:, None]
        logw, weights = torch.zeros_like(x), torch.zeros_like(x)
        for k, part in enumerate(self.families):
            if part.start == part.stop:
                continue
            mask = active[:, part]
            logits = (x[:, part] / self.cfg.pressure_temperature).masked_fill(~mask, -torch.inf)
            # Empty families are harmless; avoid inf-inf before masking.
            logits = torch.where(present[:, k, None], logits, torch.zeros_like(logits))
            log_norm = torch.logsumexp(logits, dim=1, keepdim=True)
            logw[:, part] = torch.where(mask, logits - log_norm - log_k, 0.)
            weights[:, part] = torch.where(mask, logw[:, part].exp(), 0.)
        return logw, weights

    @torch.no_grad()
    def preview(self, state, rows, operations, machines):
        """Exact analytic projection of HGNN assignment and event advancement."""
        out = state.select(rows)
        ids = out.ids
        q = torch.arange(ids.numel(), device=ids.device)
        jobs = self.op_job[ids, operations]
        start = torch.maximum(torch.maximum(out.job_ready[q, jobs], out.machine_ready[q, machines]), out.time)
        end = start + self.duration[ids, operations, machines]
        out.job_ready[q, jobs] = end
        out.machine_ready[q, machines] = end
        out.remaining[q, operations] = False
        live = torch.bmm(self.job_member[ids], out.remaining.double().unsqueeze(-1)).squeeze(-1).long()
        next_ops = torch.minimum(self.last[ids], self.last[ids] + 1 - live)
        compatible = self.eligible[ids].gather(1, next_ops[:, :, None].expand(-1, -1, self.m))
        compatible &= (live > 0)[:, :, None]
        starts = torch.maximum(out.job_ready[:, :, None], out.machine_ready[:, None])
        starts = torch.maximum(starts, out.time[:, None, None])
        next_time = starts.masked_fill(~compatible, torch.inf).flatten(1).min(-1).values
        out.time = torch.where((live > 0).any(-1), next_time, out.time)
        return out

    @torch.no_grad()
    def actor_features(self, state, eligible, next_ops):
        """Recomputed identically in rollout and evaluate; never stored in replay."""
        x, _ = self.constraints(state)
        active = self.active_constraints(state, eligible)
        logw, weights = self.log_weights(x, active)
        bid, jobs, machines = eligible.nonzero(as_tuple=True)
        features = x.new_zeros(eligible.shape + (4,))
        chunk = self.cfg.candidate_chunk_size
        for first in range(0, bid.numel(), chunk):
            rows, js, ms = bid[first:first + chunk], jobs[first:first + chunk], machines[first:first + chunk]
            post = self.preview(state, rows, next_ops[rows, js], ms)
            xp, _ = self.constraints(post)
            # The PARENT support is frozen for every counterfactual action.
            logwp, _ = self.log_weights(xp, active[rows])
            weighted = weights[rows] * (xp - x[rows])
            for d, part in enumerate(self.families):
                features[rows, js, ms, d] = weighted[:, part].sum(-1)
            kl = (weights[rows] * (logw[rows] - logwp)).sum(-1).clamp_min(0)
            features[rows, js, ms, 3] = self.cfg.pressure_temperature * kl
        count = eligible.sum((1, 2)).clamp_min(1)[:, None, None, None]
        mean = features.sum((1, 2), keepdim=True) / count
        centered = (features - mean).masked_fill(~eligible[:, :, :, None], 0.)
        scale = (centered.square().sum((1, 2), keepdim=True) / count).sqrt()
        scale = scale.clamp_min(self.cfg.actor_feature_floor)
        return (centered / scale).clamp(-self.cfg.actor_feature_clip, self.cfg.actor_feature_clip)


class RCRReplay:
    """Flatten factual snapshots in HGNN's exact [environment, time] order."""

    def __init__(self, snapshots):
        self.cores = []
        core_ids = []
        for state in snapshots:
            if state.core not in self.cores:
                self.cores.append(state.core)
            core_ids.append(self.cores.index(state.core))
        for key in ('ids', 'remaining', 'job_ready', 'machine_ready', 'time'):
            value = torch.stack([getattr(s, key) for s in snapshots], dim=0)
            setattr(self, key, value.transpose(0, 1).flatten(0, 1))
        ids = torch.tensor(core_ids, dtype=torch.long, device=self.ids.device)
        self.core_ids = ids.repeat(snapshots[0].ids.numel())

    def select(self, start, end):
        return RCRReplaySlice(self, start, end)


class RCRReplaySlice:
    def __init__(self, replay, start, end):
        self.replay, self.start, self.end = replay, start, end

    @torch.no_grad()
    def features(self, eligible, next_ops):
        replay, start, end = self.replay, self.start, self.end
        result = torch.zeros(eligible.shape + (4,), dtype=torch.float64, device=eligible.device)
        keys = ('ids', 'remaining', 'job_ready', 'machine_ready', 'time')
        for i, core in enumerate(replay.cores):
            rows = (replay.core_ids[start:end] == i).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            state = RCRState(core, *(getattr(replay, k)[start:end][rows] for k in keys))
            result[rows] = state.features(eligible[rows], next_ops[rows])
        return result


class RCREpisode:
    def __init__(self, env, options=None):
        self.config = get_rcr_config(options)
        self.core = (CapacityCore(SchedulingAdapter(env), self.config, env.proc_times_batch.device)
                     if self.config.use_actor or self.config.use_reward else None)
        self.last_delta = self.last_aux_reward = None

    def refresh(self, env):
        env.state.rcr_config = self.config
        env.state.rcr_state = self.core.snapshot(env) if self.config.use_actor else None

    def reset(self, env):
        self.last_delta = self.last_aux_reward = None
        self.refresh(env)

    @torch.no_grad()
    def augment_reward(self, env, active_ids):
        if not self.config.use_reward:
            return
        # Called only AFTER the real assignment, clock advancement and state
        # update. Include just-completed rows once, never penalize old terminals.
        state = self.core.snapshot(env).select(active_ids)
        _, delta = self.core.constraints(state)
        aux = (-self.config.reward_weight * delta /
               (self.core.counts[state.ids] * self.config.reward_scale))
        self.last_delta = torch.zeros_like(env.reward_batch)
        self.last_aux_reward = torch.zeros_like(env.reward_batch)
        self.last_delta[active_ids] = delta.to(env.reward_batch)
        self.last_aux_reward[active_ids] = aux.to(env.reward_batch)
        env.reward_batch = env.reward_batch + self.last_aux_reward
