"""
RCR-RESCHED (REINFORCE): Actor reasoning + completion-loss penalty.
"""

from copy import copy
from dataclasses import asdict, dataclass
from time import perf_counter
import json
import os

import numpy as np
import torch


@dataclass(frozen=True)
class RCRConfig:
    
    enabled: bool = True
    actor_features: bool = True
    reward_shaping: bool = True
    capacity_mode: str = 'combined'
    extra_group_count: int = 8
    window_bins: int = 8
    pressure_temperature: float = 0.1
    reward_weight: float = 0.3
    candidate_chunk_size: int = 32
    certificate_epsilon: float = 1e-7
    feature_pooling: str = 'balanced'  
    normalize_actor: bool = True
    actor_feature_floor: float = 1e-6
    actor_feature_clip: float = 5.0

    def __post_init__(self):
        for name in ('enabled', 'actor_features', 'reward_shaping', 'normalize_actor'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(name + ' must be a bool')
        if self.capacity_mode not in ('combined', 'tw', 'global'):
            raise ValueError('capacity_mode must be combined, tw, or global')
        if self.feature_pooling not in ('balanced', 'legacy'):
            raise ValueError('feature_pooling must be balanced or legacy')
        for name, minimum in (('extra_group_count', 0), ('window_bins', 1),
                              ('candidate_chunk_size', 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(name + ' must be an integer >= ' + str(minimum))
        for name in ('pressure_temperature', 'reward_weight', 'certificate_epsilon'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        if self.pressure_temperature == 0:
            raise ValueError('pressure_temperature must be positive')
        for name in ('actor_feature_floor', 'actor_feature_clip'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive')

    @property
    def use_actor(self):
        return self.enabled and self.actor_features

    @property
    def use_reward(self):
        return self.enabled and self.reward_shaping

    @classmethod
    def from_options(cls, options=None):
        if isinstance(options, cls):
            return options
        return cls(**({} if options is None else options))


def get_rcr_config(options=None):
    """Explicit model options > RCR_CONFIG environment variable > defaults."""
    if options is None:
        options = json.loads(os.environ.get('RCR_CONFIG', '{}'))
    return RCRConfig.from_options(options)


@dataclass
class AnalyticState:
    remaining: np.ndarray       # [batch, original_operations], bool
    job_ready: np.ndarray       # [batch, jobs], including finished jobs
    machine_ready: np.ndarray   # [batch, machines], append-only tails
    time: np.ndarray            # [batch], earliest legal start frontier (Actor pooling only)


@dataclass
class TorchAnalyticState:
    remaining: torch.Tensor      # [batch, original_operations], bool
    job_ready: torch.Tensor      # [batch, jobs]
    machine_ready: torch.Tensor  # [batch, machines]
    time: torch.Tensor           # [batch]


@dataclass
class Witness:
    actions: np.ndarray         # [batch, decisions, 2], native sequence/machine
    operations: np.ndarray      # [batch, N], original operation ids
    machines: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    makespan: np.ndarray


class SchedulingAdapter:
    """Exact analytic transition for the supplied SchedulingEnvironment.py."""

    def __init__(self, env):
        p = env.problem
        self.duration = np.asarray(p.duration, dtype=np.float64).copy()
        self.eligible = self.duration > 0
        self.batch_size, self.n, self.m = self.duration.shape
        if self.n < 2:
            raise ValueError('The native environment requires at least two operations.')
        if (not np.isfinite(self.duration).all() or (self.duration < 0).any()
                or not self.eligible.any(-1).all()):
            raise ValueError('RCR requires finite nonnegative times and no padded operations.')
        if not np.equal(self.duration, np.floor(self.duration)).all():
            raise ValueError('Native start/finish arrays store integers; use integer times.')
        self.pmin = np.where(self.eligible, self.duration, np.inf).min(-1)
        self.op_job = np.empty((self.batch_size, self.n), dtype=np.int64)
        job_lists = [np.unique(row) for row in p.job_idx]
        self.j = max(map(len, job_lists))
        self.job_valid = np.zeros((self.batch_size, self.j), dtype=bool)
        self.job_member = np.zeros((self.batch_size, self.j, self.n), dtype=np.float64)
        self.prefix = np.zeros_like(self.pmin)
        self.suffix = np.zeros_like(self.pmin)
        # The native makespan update assumes a chain for each job. Fail clearly
        # for a different problem rather than produce an unsound certificate.
        expected_dep = np.zeros_like(p.dependency, dtype=bool)
        for b, jobs in enumerate(job_lists):
            for j, job in enumerate(jobs):
                ops = np.flatnonzero(p.job_idx[b] == job)
                if not np.array_equal(p.operation_idx[b, ops], np.arange(len(ops))):
                    raise ValueError('Operations must be ordered 0..n_j-1 within each job.')
                expected_dep[b, ops[:-1], ops[1:]] = True
                self.op_job[b, ops] = j
                self.job_valid[b, j] = True
                self.job_member[b, j, ops] = 1
                work = self.pmin[b, ops]
                self.prefix[b, ops] = np.cumsum(work) - work
                self.suffix[b, ops] = work.sum() - np.cumsum(work)
        if not np.array_equal(expected_dep, p.dependency):
            raise ValueError('This RCR adapter supports the native per-job chain dependencies.')
        self.total_work = np.einsum('bjn,bn->bj', self.job_member, self.pmin)
        self._torch_cache = {}

    def export_state(self, env):
        finish = np.asarray(env.state.finish_time).max(-1)
        remaining = finish < 0
        job_ready = np.max(np.where(self.job_member > 0, finish[:, None, :], 0),
                           axis=-1).clip(min=0).astype(np.float64)
        machine_ready = np.array(env.state.m_AT, dtype=np.float64, copy=True)

        # RESCHED is append-only but not an event-clock policy: actions may be
        # selected in a non-chronological construction order.  We therefore do
        # NOT advance the certificate by a global clock.  This frontier is used
        # only to suppress already-past local windows in Actor pooling.
        if env.state.action_mask.size and env.state.action_mask.any():
            pair_start = np.maximum(env.state.o_AT[:, :, None],
                                    env.state.m_AT[:, None, :])
            frontier = np.where(env.state.action_mask, pair_start, np.inf).min((1, 2))
        else:
            frontier = job_ready.max(-1)
        frontier = np.where(np.isfinite(frontier), frontier, job_ready.max(-1))

        return AnalyticState(remaining=remaining,
                             job_ready=job_ready,
                             machine_ready=machine_ready,
                             time=frontier.astype(np.float64))

    def preview(self, state, batch_ids, operations, machines):
        """Selected legal actions, vectorized over a bounded candidate chunk."""
        ids = np.asarray(batch_ids, dtype=np.int64)
        rows = np.arange(len(ids))
        result = AnalyticState(state.remaining[ids].copy(), state.job_ready[ids].copy(),
                               state.machine_ready[ids].copy(), state.time[ids].copy())

        def assign(which, ops, mas):
            instance = ids[which]
            jobs = self.op_job[instance, ops]
            ends = np.maximum(result.job_ready[which, jobs],
                              result.machine_ready[which, mas]) + self.duration[instance, ops, mas]
            result.job_ready[which, jobs] = ends
            result.machine_ready[which, mas] = ends
            result.remaining[which, ops] = False

        assign(rows, operations, machines)
        auto = np.flatnonzero(result.remaining.sum(-1) == 1)
        if len(auto):
            last_ops = result.remaining[auto].argmax(-1)
            last_p = self.duration[ids[auto], last_ops]
            # Intentionally match the ORIGINAL selector: it uses m_AT + p,
            # without job readiness. The actual start still includes readiness.
            last_m = (result.machine_ready[auto] +
                      np.where(last_p > 0, last_p, np.inf)).argmin(-1)
            assign(auto, last_ops, last_m)
        return result

    def _torch_static(self, device, dtype=torch.float32):
        device = torch.device(device)
        key = (str(device), dtype)
        cached = self._torch_cache.get(key)
        if cached is None:
            cached = dict(
                duration=torch.as_tensor(self.duration, dtype=dtype, device=device),
                eligible=torch.as_tensor(self.eligible, dtype=torch.bool, device=device),
                pmin=torch.as_tensor(self.pmin, dtype=dtype, device=device),
                op_job=torch.as_tensor(self.op_job, dtype=torch.long, device=device),
                job_valid=torch.as_tensor(self.job_valid, dtype=torch.bool, device=device),
                job_member=torch.as_tensor(self.job_member, dtype=dtype, device=device),
                prefix=torch.as_tensor(self.prefix, dtype=dtype, device=device),
                suffix=torch.as_tensor(self.suffix, dtype=dtype, device=device),
                total_work=torch.as_tensor(self.total_work, dtype=dtype, device=device),
            )
            self._torch_cache[key] = cached
        return cached

    @torch.no_grad()
    def export_state_torch(self, env, device, dtype=torch.float32,
                           machine_ready=None, o_at=None, action_mask=None):
        """Export the authoritative CPU environment state once to Torch.
        """
        device = torch.device(device)
        t = self._torch_static(device, dtype)
        finish = torch.as_tensor(env.state.finish_time, dtype=dtype, device=device).amax(-1)
        remaining = finish < 0
        job_ready = torch.where(
            t['job_member'] > 0,
            finish[:, None, :],
            torch.zeros((), dtype=dtype, device=device),
        ).amax(-1).clamp_min_(0)
        if machine_ready is None:
            machine_ready = torch.as_tensor(env.state.m_AT, dtype=dtype, device=device)
        else:
            machine_ready = machine_ready.to(device=device, dtype=dtype, non_blocking=True)

        if action_mask is None:
            action_mask = torch.as_tensor(env.state.action_mask, dtype=torch.bool, device=device)
        else:
            action_mask = action_mask.to(device=device, dtype=torch.bool, non_blocking=True)
        if action_mask.numel():
            if o_at is None:
                o_at = torch.as_tensor(env.state.o_AT, dtype=dtype, device=device)
            else:
                o_at = o_at.to(device=device, dtype=dtype, non_blocking=True)
            pair_start = torch.maximum(o_at[:, :, None], machine_ready[:, None, :])
            frontier = pair_start.masked_fill(~action_mask, float('inf')).amin(dim=(1, 2))
        else:
            frontier = job_ready.amax(-1)
        fallback = job_ready.amax(-1)
        frontier = torch.where(torch.isfinite(frontier), frontier, fallback)
        return TorchAnalyticState(remaining, job_ready, machine_ready, frontier)

    @torch.no_grad()
    def preview_torch(self, state, batch_ids, operations, machines):
        """GPU/CPU Torch preview vectorized over a candidate batch."""
        ids = batch_ids.to(dtype=torch.long)
        operations = operations.to(dtype=torch.long)
        machines = machines.to(dtype=torch.long)
        q = ids.numel()
        device = ids.device
        dtype = state.job_ready.dtype
        t = self._torch_static(device, dtype)
        rows = torch.arange(q, device=device)
        result = TorchAnalyticState(
            state.remaining.index_select(0, ids).clone(),
            state.job_ready.index_select(0, ids).clone(),
            state.machine_ready.index_select(0, ids).clone(),
            state.time.index_select(0, ids).clone(),
        )

        def assign(which, ops, mas):
            instance = ids.index_select(0, which)
            jobs = t['op_job'][instance, ops]
            ends = torch.maximum(result.job_ready[which, jobs],
                                 result.machine_ready[which, mas])
            ends = ends + t['duration'][instance, ops, mas]
            result.job_ready[which, jobs] = ends
            result.machine_ready[which, mas] = ends
            result.remaining[which, ops] = False

        assign(rows, operations, machines)
        auto = torch.nonzero(result.remaining.sum(-1) == 1, as_tuple=False).flatten()
        if auto.numel():
            last_ops = result.remaining.index_select(0, auto).to(torch.int64).argmax(-1)
            instance = ids.index_select(0, auto)
            last_p = t['duration'][instance, last_ops]
            inf = torch.full_like(last_p, float('inf'))
            selector = result.machine_ready.index_select(0, auto) + torch.where(last_p > 0, last_p, inf)
            last_m = selector.argmin(-1)
            assign(auto, last_ops, last_m)
        return result

    def make_witness(self, env):
        """Two deterministic native rollouts; no generator/RNG calls or mutation.

        SPT: shortest eligible processing time, then earliest completion.
        MWKR: largest remaining pmin job work, then earliest completion.
        Further ties use the native sequence index and machine index.
        """
        witnesses = []
        for rule in ('spt', 'mwkr'):
            reference = copy(env)
            reference.state = type(env.state)()
            reference.solution = type(env.solution)()
            reference.reset_state()
            actions = []
            done = False
            while not done:
                state = reference.state
                remaining = np.where(state.duration > 0, state.duration, np.inf).min(-1)
                seq, machine = [], []
                for b in range(self.batch_size):
                    o, m = np.nonzero(state.action_mask[b])
                    proc = state.duration[b, o, m]
                    end = np.maximum(state.o_AT[b, o], state.m_AT[b, m]) + proc
                    if rule == 'spt':
                        primary = proc
                    else:
                        work = {j: remaining[b, state.job_idx[b] == j].sum()
                                for j in np.unique(state.job_idx[b])}
                        primary = -np.array([work[j] for j in state.job_idx[b, o]])
                    choice = np.lexsort((m, o, end, primary))[0]
                    seq.append(o[choice])
                    machine.append(m[choice])
                action = np.column_stack((seq, machine))
                actions.append(action)
                _, _, done = reference.step((action[:, 0], action[:, 1]))
            sol = reference.solution
            ops, mas = np.array(sol.jo_idx).T, np.array(sol.machine_idx)
            rows = np.arange(self.batch_size)[:, None]
            # The supplied _makespan updates dense state calendars but leaves
            # Solution.start_time/end_time empty. Read the authoritative state.
            witness = Witness(np.stack(actions, axis=1), ops, mas,
                              reference.state.start_time[rows, ops, mas],
                              reference.state.finish_time[rows, ops, mas],
                              reference.state.finish_time.max((1, 2)))
            self.validate_witness(witness)
            witnesses.append(witness)
        first, second = witnesses
        choose = first.makespan <= second.makespan
        fields = {}
        for name in Witness.__dataclass_fields__:
            left, right = getattr(first, name), getattr(second, name)
            mask = choose.reshape((self.batch_size,) + (1,) * (left.ndim - 1))
            fields[name] = np.where(mask, left, right)
        return Witness(**fields)

    def validate_witness(self, witness):
        for b in range(self.batch_size):
            ops, mas = witness.operations[b], witness.machines[b]
            starts, ends = witness.starts[b], witness.ends[b]
            if not np.array_equal(np.sort(ops), np.arange(self.n)):
                raise ValueError('Reference schedule did not assign every operation once.')
            if ((starts < 0).any() or not self.eligible[b, ops, mas].all()
                    or not np.array_equal(ends - starts, self.duration[b, ops, mas])):
                raise ValueError('Invalid reference processing times/eligibility.')
            order = np.argsort(ops)
            for j in range(self.j):
                job_ops = np.flatnonzero(self.job_member[b, j])
                if (starts[order[job_ops[1:]]] < ends[order[job_ops[:-1]]]).any():
                    raise ValueError('Reference schedule violates job precedence.')
            for m in range(self.m):
                which = np.flatnonzero(mas == m)
                which = which[np.argsort(starts[which], kind='stable')]
                if (starts[which[1:]] < ends[which[:-1]]).any():
                    raise ValueError('Reference schedule has overlapping machine use.')
            if ends.max() != witness.makespan[b]:
                raise ValueError('Reference makespan mismatch.')


class RCRCore:
    """Fixed groups/windows and analytic constraints; no torch or optimizer."""

    feature_names = ('job_path_pressure', 'global_capacity_pressure',
                     'local_capacity_pressure', 'bottleneck_redistribution')

    def __init__(self, adapter, horizon, config):
        self.adapter, self.config = adapter, config
        self.horizon = np.asarray(horizon, dtype=np.float64).copy()
        if (self.horizon <= 0).any() or not np.isfinite(self.horizon).all():
            raise ValueError('H must be a finite positive feasible makespan.')
        a = adapter
        groups = []
        for b in range(a.batch_size):
            base = [tuple([m]) for m in range(a.m)]
            all_m = tuple(range(a.m))
            if all_m not in base:
                base.append(all_m)
            candidates = {tuple(np.flatnonzero(row)) for row in a.eligible[b]}
            candidates = [s for s in candidates if 1 < len(s) < a.m]

            def rank(s):
                mask = np.zeros(a.m, dtype=bool)
                mask[list(s)] = True
                included = (~a.eligible[b] | mask).all(-1)
                return (-a.pmin[b, included].sum() / len(s), s)

            if config.capacity_mode != 'tw':
                base += sorted(candidates, key=rank)[:config.extra_group_count]
            groups.append(base)

        self.groups = groups
        self.g = max(map(len, groups))
        self.group_valid = np.zeros((a.batch_size, self.g), dtype=bool)
        self.machine_member = np.zeros((a.batch_size, self.g, a.m), dtype=np.float64)
        for b, sets in enumerate(groups):
            for g, s in enumerate(sets):
                self.group_valid[b, g] = True
                self.machine_member[b, g, list(s)] = 1

        self.op_member = ((~a.eligible[:, None, :, :] |
                           self.machine_member[:, :, None, :].astype(bool)).all(-1)
                          & self.group_valid[:, :, None]).astype(np.float64)
        self.group_size = self.machine_member.sum(-1).clip(min=1)

        bins = config.window_bins
        fractions = [(0., 1.)]
        if config.capacity_mode != 'global':
            fractions += [(i / bins, j / bins) for i in range(bins)
                          for j in range(i + 1, bins + 1) if (i, j) != (0, bins)]
        self.windows = self.horizon[:, None, None] * np.asarray(fractions)[None, :, :]
        self.w = len(fractions)
        self.job_end = a.j
        self.global_end = a.j + self.g
        self.valid = np.concatenate((a.job_valid, self.group_valid,
                                     np.repeat(self.group_valid, self.w - 1, axis=1)), axis=1)
        self._torch_cache = {}

    @staticmethod
    def mandatory_work(pmin, earliest, latest, left, right):
        return np.maximum(0., np.minimum(np.minimum(pmin, right - left),
                          np.minimum(earliest + pmin - left, right - latest + pmin)))

    def constraints(self, state, batch_ids=None):
        """Return dimensionless pressure x and completion-loss Delta."""
        a = self.adapter
        ids = np.arange(a.batch_size) if batch_ids is None else np.asarray(batch_ids)
        q = len(ids)
        remaining = state.remaining
        pmin = a.pmin[ids]

        work = np.einsum('qjn,qn->qj', a.job_member[ids], remaining * pmin)
        completed_work = a.total_work[ids] - work
        jobs = a.op_job[ids]
        rows = np.arange(q)[:, None]

        # No artificial global time is inserted here. RESCHED may construct
        # independent machine tails out of chronological order.
        earliest = (state.job_ready[rows, jobs] + a.prefix[ids]
                    - completed_work[rows, jobs])
        latest = self.horizon[ids, None] - a.suffix[ids]

        left, right = self.windows[ids, :, 0], self.windows[ids, :, 1]
        mandatory = self.mandatory_work(pmin[:, :, None], earliest[:, :, None],
                                       latest[:, :, None], left[:, None, :], right[:, None, :])
        mandatory *= remaining[:, :, None]
        demand = np.matmul(self.op_member[ids], mandatory)

        free = np.maximum(0., right[:, None, :] -
                          np.maximum(left[:, None, :], state.machine_ready[:, :, None]))
        capacity = np.matmul(self.machine_member[ids], free)
        violation = demand - capacity

        h = self.horizon[ids, None]
        job_x = (state.job_ready + work - h) / h
        global_x = violation[:, :, 0] / (h * self.group_size[ids])
        n_group = np.einsum('qgn,qn->qg', self.op_member[ids], remaining).clip(min=1)
        local_x = violation[:, :, 1:] / (h[:, :, None] * n_group[:, :, None])

        x = np.concatenate((job_x, global_x, local_x.reshape(q, -1)), axis=1)
        x = np.where(self.valid[ids], x, 0.)
        largest = np.where(self.valid[ids], x, -np.inf).max(-1)
        delta = self.horizon[ids] * np.maximum(
            0., largest - self.config.certificate_epsilon)
        return x, delta

    @property
    def families(self):
        return (slice(0, self.job_end),
                slice(self.job_end, self.global_end),
                slice(self.global_end, self.valid.shape[1]))

    def active_constraints(self, state):
       
        a = self.adapter
        job_live = np.einsum('bjn,bn->bj', a.job_member, state.remaining) > 0
        group_live = ((np.einsum('bgn,bn->bg', self.op_member, state.remaining) > 0)
                      & self.group_valid)
        window_live = self.windows[:, 1:, 1] > state.time[:, None]
        local_live = group_live[:, :, None] & window_live[:, None, :]
        return self.valid & np.concatenate(
            (job_live, group_live, local_live.reshape(a.batch_size, -1)), axis=1)

    def log_weights(self, x, batch_ids, active=None):
        valid = self.valid[batch_ids] if active is None else active & self.valid[batch_ids]

        if self.config.feature_pooling == 'legacy':
            z = np.where(valid, x / self.config.pressure_temperature, -np.inf)
            maximum = z.max(-1, keepdims=True)
            logsum = maximum + np.log(np.exp(z - maximum).sum(-1, keepdims=True))
            logw = np.where(valid, z - logsum, 0.)
            return logw, np.where(valid, np.exp(logw), 0.)

        # Equal total mass per nonempty family, independent of how many
        # job/group/window constraints that family contains.
        logw, w = np.zeros_like(x), np.zeros_like(x)
        present = np.stack([valid[:, part].any(-1) for part in self.families], axis=1)
        family_count = present.sum(-1).clip(min=1)

        for k, part in enumerate(self.families):
            if part.start == part.stop:
                continue
            mask = valid[:, part]
            z = np.where(mask, x[:, part] / self.config.pressure_temperature, -np.inf)
            maximum = np.where(present[:, k], z.max(-1), 0.)[:, None]
            total = np.exp(z - maximum).sum(-1, keepdims=True)
            logsum = maximum + np.log(total.clip(min=1e-300))
            logw[:, part] = np.where(
                mask, z - logsum - np.log(family_count[:, None]), 0.)
            w[:, part] = np.where(mask, np.exp(logw[:, part]), 0.)
        return logw, w

    def _torch_static(self, device, dtype=torch.float32):
        device = torch.device(device)
        key = (str(device), dtype)
        cached = self._torch_cache.get(key)
        if cached is None:
            cached = dict(
                horizon=torch.as_tensor(self.horizon, dtype=dtype, device=device),
                group_valid=torch.as_tensor(self.group_valid, dtype=torch.bool, device=device),
                machine_member=torch.as_tensor(self.machine_member, dtype=dtype, device=device),
                op_member=torch.as_tensor(self.op_member, dtype=dtype, device=device),
                group_size=torch.as_tensor(self.group_size, dtype=dtype, device=device),
                windows=torch.as_tensor(self.windows, dtype=dtype, device=device),
                valid=torch.as_tensor(self.valid, dtype=torch.bool, device=device),
            )
            self._torch_cache[key] = cached
        return cached

    @staticmethod
    def mandatory_work_torch(pmin, earliest, latest, left, right):
        zero = torch.zeros((), dtype=pmin.dtype, device=pmin.device)
        return torch.maximum(zero, torch.minimum(torch.minimum(pmin, right - left),
                             torch.minimum(earliest + pmin - left,
                                           right - latest + pmin)))

    @torch.no_grad()
    def constraints_torch(self, state, batch_ids=None):
        
        a = self.adapter
        device = state.job_ready.device
        dtype = state.job_ready.dtype
        at = a._torch_static(device, dtype)
        ct = self._torch_static(device, dtype)

        if batch_ids is None:
            ids = None
            q = a.batch_size
            pmin = at['pmin']
            job_member = at['job_member']
            total_work = at['total_work']
            jobs = at['op_job']
            prefix = at['prefix']
            suffix = at['suffix']
            horizon = ct['horizon']
            windows = ct['windows']
            op_member = ct['op_member']
            machine_member = ct['machine_member']
            group_size = ct['group_size']
            valid = ct['valid']
        else:
            ids = batch_ids.to(device=device, dtype=torch.long)
            q = ids.numel()
            pmin = at['pmin'].index_select(0, ids)
            job_member = at['job_member'].index_select(0, ids)
            total_work = at['total_work'].index_select(0, ids)
            jobs = at['op_job'].index_select(0, ids)
            prefix = at['prefix'].index_select(0, ids)
            suffix = at['suffix'].index_select(0, ids)
            horizon = ct['horizon'].index_select(0, ids)
            windows = ct['windows'].index_select(0, ids)
            op_member = ct['op_member'].index_select(0, ids)
            machine_member = ct['machine_member'].index_select(0, ids)
            group_size = ct['group_size'].index_select(0, ids)
            valid = ct['valid'].index_select(0, ids)

        rem = state.remaining.to(dtype)
        weighted_remaining = rem * pmin
        work = torch.bmm(job_member, weighted_remaining.unsqueeze(-1)).squeeze(-1)
        completed_work = total_work - work
        job_ready_by_op = state.job_ready.gather(1, jobs)
        completed_by_op = completed_work.gather(1, jobs)
        earliest = job_ready_by_op + prefix - completed_by_op
        latest = horizon[:, None] - suffix

        left, right = windows[:, :, 0], windows[:, :, 1]
        mandatory = self.mandatory_work_torch(
            pmin[:, :, None], earliest[:, :, None], latest[:, :, None],
            left[:, None, :], right[:, None, :])
        mandatory.mul_(rem[:, :, None])
        demand = torch.matmul(op_member, mandatory)

        free = torch.clamp_min(
            right[:, None, :] - torch.maximum(left[:, None, :],
                                               state.machine_ready[:, :, None]), 0.)
        capacity = torch.matmul(machine_member, free)
        violation = demand - capacity

        h = horizon[:, None]
        job_x = (state.job_ready + work - h) / h
        global_x = violation[:, :, 0] / (h * group_size)
        n_group = torch.bmm(op_member, rem.unsqueeze(-1)).squeeze(-1).clamp_min_(1.)
        local_x = violation[:, :, 1:] / (h[:, :, None] * n_group[:, :, None])

        x = torch.cat((job_x, global_x, local_x.reshape(q, -1)), dim=1)
        x = torch.where(valid, x, torch.zeros((), dtype=dtype, device=device))
        largest = x.masked_fill(~valid, float('-inf')).amax(-1)
        delta = horizon * torch.clamp_min(
            largest - self.config.certificate_epsilon, 0.)
        return x, delta

    @torch.no_grad()
    def active_constraints_torch(self, state):
        a = self.adapter
        device = state.job_ready.device
        dtype = state.job_ready.dtype
        at = a._torch_static(device, dtype)
        ct = self._torch_static(device, dtype)
        rem = state.remaining.to(dtype)
        job_live = torch.bmm(at['job_member'], rem.unsqueeze(-1)).squeeze(-1) > 0
        group_live = ((torch.bmm(ct['op_member'], rem.unsqueeze(-1)).squeeze(-1) > 0)
                      & ct['group_valid'])
        window_live = ct['windows'][:, 1:, 1] > state.time[:, None]
        local_live = group_live[:, :, None] & window_live[:, None, :]
        active = torch.cat((job_live, group_live,
                            local_live.reshape(a.batch_size, -1)), dim=1)
        return ct['valid'] & active

    @torch.no_grad()
    def log_weights_torch(self, x, batch_ids, active=None):
        device, dtype = x.device, x.dtype
        ct = self._torch_static(device, dtype)
        ids = batch_ids.to(device=device, dtype=torch.long)
        valid = ct['valid'].index_select(0, ids) if active is None else (
            active & ct['valid'].index_select(0, ids))
        tau = self.config.pressure_temperature

        if self.config.feature_pooling == 'legacy':
            z = (x / tau).masked_fill(~valid, float('-inf'))
            logsum = torch.logsumexp(z, dim=-1, keepdim=True)
            logw = torch.where(valid, z - logsum, torch.zeros_like(z))
            return logw, torch.where(valid, logw.exp(), torch.zeros_like(logw))

        logw = torch.zeros_like(x)
        w = torch.zeros_like(x)
        present = torch.stack([valid[:, part].any(-1) for part in self.families], dim=1)
        family_count = present.sum(-1).clamp_min(1).to(dtype)
        log_family_count = family_count.log()[:, None]

        for part in self.families:  # exactly three fixed families; no candidate loop
            if part.start == part.stop:
                continue
            mask = valid[:, part]
            z = (x[:, part] / tau).masked_fill(~mask, float('-inf'))
            logsum = torch.logsumexp(z, dim=-1, keepdim=True)
            values = z - logsum - log_family_count
            values = torch.where(mask, values, torch.zeros_like(values))
            logw[:, part] = values
            w[:, part] = torch.where(mask, values.exp(), torch.zeros_like(values))
        return logw, w

    def _gpu_chunk_size(self, candidate_count, device):
        
        if candidate_count <= 0:
            return 1
        if torch.device(device).type != 'cuda':
            return max(1, self.config.candidate_chunk_size)
        # Roughly cap the largest certificate temporaries to ~64 MB in fp32.
        per_candidate = max(1, self.adapter.n * self.w + 3 * self.g * self.w
                            + 4 * self.valid.shape[1])
        adaptive = max(128, min(4096, 16_000_000 // per_candidate))
        return min(candidate_count, max(self.config.candidate_chunk_size, adaptive))

    @torch.no_grad()
    def raw_actor_features_torch(self, state, action_mask):
        a, cfg = self.adapter, self.config
        action_mask = action_mask.to(device=state.job_ready.device, dtype=torch.bool)
        x, _ = self.constraints_torch(state)
        active = (self.active_constraints_torch(state)
                  if cfg.feature_pooling == 'balanced'
                  else self._torch_static(x.device, x.dtype)['valid'])
        all_ids = torch.arange(a.batch_size, device=x.device)
        logw, weights = self.log_weights_torch(x, all_ids, active)

        remaining_count = state.remaining.sum(-1)
        if state.job_ready.device.type == 'cpu' and not bool(torch.all(remaining_count == action_mask.shape[1])):
            raise ValueError('Native action mask and original-operation mapping disagree.')
        remaining_ids = torch.nonzero(state.remaining, as_tuple=False)[:, 1].reshape(
            a.batch_size, action_mask.shape[1])

        batch_ids, sequences, machines = torch.nonzero(action_mask, as_tuple=True)
        features = torch.zeros(action_mask.shape + (4,), dtype=x.dtype, device=x.device)
        count = batch_ids.numel()
        chunk = self._gpu_chunk_size(count, x.device)

        for start in range(0, count, chunk):
            sl = slice(start, min(start + chunk, count))
            ids, seq, mas = batch_ids[sl], sequences[sl], machines[sl]
            ops = remaining_ids[ids, seq]
            parent_x = x.index_select(0, ids)
            parent_w = weights.index_select(0, ids)
            parent_logw = logw.index_select(0, ids)
            parent_active = active.index_select(0, ids)
            post = a.preview_torch(state, ids, ops, mas)
            post_x, _ = self.constraints_torch(post, ids)
            post_logw, _ = self.log_weights_torch(post_x, ids, parent_active)
            change = parent_w * (post_x - parent_x)
            f0 = change[:, :self.job_end].sum(-1)
            f1 = change[:, self.job_end:self.global_end].sum(-1)
            f2 = change[:, self.global_end:].sum(-1)
            kl = (parent_w * (parent_logw - post_logw)).sum(-1)
            f3 = cfg.pressure_temperature * torch.clamp_min(kl, 0.)
            features[ids, seq, mas] = torch.stack((f0, f1, f2, f3), dim=-1)
        return features

    @torch.no_grad()
    def actor_features_torch(self, state, action_mask):
        raw = self.raw_actor_features_torch(state, action_mask)
        return condition_actor_features_torch(raw, action_mask, self.config)

    def raw_actor_features(self, state, action_mask):
        a, cfg = self.adapter, self.config
        x, _ = self.constraints(state)
        active = self.active_constraints(state) if cfg.feature_pooling == 'balanced' else self.valid
        logw, weights = self.log_weights(x, np.arange(a.batch_size), active)

        remaining_count = state.remaining.sum(-1)
        if not np.all(remaining_count == action_mask.shape[1]):
            raise ValueError('Native action mask and original-operation mapping disagree.')
        remaining_ids = np.nonzero(state.remaining)[1].reshape(
            a.batch_size, action_mask.shape[1])

        batch_ids, sequences, machines = np.nonzero(action_mask)
        features = np.zeros(action_mask.shape + (4,), dtype=np.float64)

        for start in range(0, len(batch_ids), cfg.candidate_chunk_size):
            sl = slice(start, start + cfg.candidate_chunk_size)
            ids, seq, mas = batch_ids[sl], sequences[sl], machines[sl]
            post = a.preview(state, ids, remaining_ids[ids, seq], mas)
            post_x, _ = self.constraints(post, ids)

            # Freeze the parent active mask for all counterfactual successors.
            post_logw, _ = self.log_weights(post_x, ids, active[ids])
            change = weights[ids] * (post_x - x[ids])
            features[ids, seq, mas, 0] = change[:, :self.job_end].sum(-1)
            features[ids, seq, mas, 1] = change[:, self.job_end:self.global_end].sum(-1)
            features[ids, seq, mas, 2] = change[:, self.global_end:].sum(-1)
            kl = (weights[ids] * (logw[ids] - post_logw)).sum(-1)
            features[ids, seq, mas, 3] = (
                cfg.pressure_temperature * np.maximum(kl, 0.))

        return features

    def actor_features(self, state, action_mask):
        raw = self.raw_actor_features(state, action_mask)
        return condition_actor_features(raw, action_mask, self.config)


def condition_actor_features(raw, action_mask, config):
    """Condition only neural Actor inputs; keep raw certificates untouched."""
    if not np.isfinite(raw).all():
        raise FloatingPointError('RCR Actor features are not finite.')
    if not config.normalize_actor:
        return np.where(action_mask[..., None], raw, 0.)

    count = action_mask.sum((1, 2)).clip(min=1)[:, None, None, None]
    mask = action_mask[..., None]
    mean = np.where(mask, raw, 0.).sum((1, 2), keepdims=True) / count
    centered = np.where(mask, raw - mean, 0.)
    variance = np.square(centered).sum((1, 2), keepdims=True) / count
    scale = np.maximum(np.sqrt(variance), config.actor_feature_floor)
    return np.clip(centered / scale,
                   -config.actor_feature_clip,
                   config.actor_feature_clip)


@torch.no_grad()
def condition_actor_features_torch(raw, action_mask, config):
    """Torch/GPU equivalent of condition_actor_features()."""
    
    if raw.device.type == 'cpu' and not bool(torch.isfinite(raw).all()):
        raise FloatingPointError('RCR Actor features are not finite.')
    mask3 = action_mask.to(device=raw.device, dtype=torch.bool)
    mask = mask3[..., None]
    if not config.normalize_actor:
        return torch.where(mask, raw, torch.zeros_like(raw))
    count = mask3.sum((1, 2)).clamp_min(1).to(raw.dtype)[:, None, None, None]
    mean = torch.where(mask, raw, torch.zeros_like(raw)).sum((1, 2), keepdim=True) / count
    centered = torch.where(mask, raw - mean, torch.zeros_like(raw))
    variance = centered.square().sum((1, 2), keepdim=True) / count
    scale = variance.sqrt().clamp_min(config.actor_feature_floor)
    return (centered / scale).clamp(-config.actor_feature_clip,
                                    config.actor_feature_clip)


class RCREpisode:
    def __init__(self, env, config):
        start = perf_counter()
        self.config = config
        self.adapter = SchedulingAdapter(env)
        self.witness = self.adapter.make_witness(env)
        self.core = RCRCore(self.adapter, self.witness.makespan, config)
        self.preprocess_seconds = perf_counter() - start
        self.reset_statistics()

    def reset_statistics(self):
        self.feature_seconds = 0.
        self.reward_seconds = 0.
        self._torch_state_cache = {}
        b = self.adapter.batch_size
        self.last_delta = np.zeros(b, dtype=np.float64)
        self.last_base_reward = np.zeros(b, dtype=np.float64)
        self.last_aux_reward = np.zeros(b, dtype=np.float64)
        self.last_total_reward = np.zeros(b, dtype=np.float64)

    def actor_features(self, env):
        start = perf_counter()
        result = self.core.actor_features(
            self.adapter.export_state(env), env.state.action_mask)
        self.feature_seconds += perf_counter() - start
        return result

    def _torch_state_key(self, env, device, dtype):
        return (str(torch.device(device)), dtype, len(env.solution.jo_idx))

    @torch.no_grad()
    def _get_torch_state(self, env, device, dtype, machine_ready=None, o_at=None, action_mask=None):
        key = self._torch_state_key(env, device, dtype)
        cached = self._torch_state_cache.get(key)
        if cached is not None:
            return cached
        state = self.adapter.export_state_torch(
            env, device, dtype, machine_ready=machine_ready, o_at=o_at, action_mask=action_mask)
        # Only the current construction step is reusable; discard stale states.
        self._torch_state_cache = {key: state}
        return state

    @torch.no_grad()
    def actor_features_torch(self, env, device, dtype=torch.float32,
                             action_mask=None, machine_ready=None, o_at=None):
        start = perf_counter()
        state = self._get_torch_state(
            env, device, dtype, machine_ready=machine_ready, o_at=o_at, action_mask=action_mask)
        if action_mask is None:
            action_mask = torch.as_tensor(env.state.action_mask, dtype=torch.bool, device=device)
        result = self.core.actor_features_torch(state, action_mask)
        self.feature_seconds += perf_counter() - start
        return result

    def compose_reward(self, env, base_reward):
        start = perf_counter()
        base = np.asarray(base_reward, dtype=np.float64)
        if base.shape != (self.adapter.batch_size,):
            raise ValueError('base_reward must have shape [batch].')

        _, delta = self.core.constraints(self.adapter.export_state(env))
        aux = -self.config.reward_weight * delta / float(self.adapter.n)
        total = base + aux

        if not (np.isfinite(delta).all() and np.isfinite(aux).all()
                and np.isfinite(total).all()):
            raise FloatingPointError('RCR reward contains non-finite values.')

        self.last_delta = delta.copy()
        self.last_base_reward = base.copy()
        self.last_aux_reward = aux.copy()
        self.last_total_reward = total.copy()
        self.reward_seconds += perf_counter() - start
        return total

    @torch.no_grad()
    def compose_reward_torch(self, env, base_reward, device, dtype=torch.float32):
        """GPU-vectorized completion-loss reward on the REAL successor state."""
        start = perf_counter()
        base = torch.as_tensor(base_reward, dtype=dtype, device=device)
        if tuple(base.shape) != (self.adapter.batch_size,):
            raise ValueError('base_reward must have shape [batch].')
        state = self._get_torch_state(env, device, dtype)
        _, delta = self.core.constraints_torch(state)
        aux = -self.config.reward_weight * delta / float(self.adapter.n)
        total = base + aux
        if total.device.type == 'cpu' and not bool(torch.isfinite(total).all()):
            raise FloatingPointError('RCR reward contains non-finite values.')
        self.last_delta = delta.detach()
        self.last_base_reward = base.detach()
        self.last_aux_reward = aux.detach()
        self.last_total_reward = total.detach()
        self.reward_seconds += perf_counter() - start
        return total

    @staticmethod
    def _tolist(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return np.asarray(value).tolist()

    def statistics(self):
        return dict(config=asdict(self.config),
                    horizon=self.witness.makespan.tolist(),
                    preprocess_seconds=self.preprocess_seconds,
                    feature_seconds=self.feature_seconds,
                    reward_seconds=self.reward_seconds,
                    last_delta=self._tolist(self.last_delta),
                    last_base_reward=self._tolist(self.last_base_reward),
                    last_aux_reward=self._tolist(self.last_aux_reward),
                    last_total_reward=self._tolist(self.last_total_reward))


def get_rcr(env, config):
    """Episode-local cache invalidated by reset/load, even if sizes are equal."""
    token = env.state.finish_time
    if getattr(env, '_rcr_reset_token', None) is not token:
        env._rcr_reset_token = token
        env._rcr_episodes = {}
    if config not in env._rcr_episodes:
        env._rcr_episodes[config] = RCREpisode(env, config)
    return env._rcr_episodes[config]


class ModelInput(tuple):
    """Keep the original eight-tensor interface and the unchanged evaluator.
    """
    def __new__(cls, tensors, env, rcr_dynamic=None):
        result = super().__new__(cls, tensors)
        result._env = env
        result._token = env.state.finish_time
        result._step = len(env.solution.jo_idx)
        result._features = {}
        result._rcr_dynamic = rcr_dynamic
        return result

    def device_tensors(self, device):
        device = torch.device(device)
        key = ('tensors', str(device))
        if key not in self._features:
            moved = tuple(t if t.device == device else t.to(device, non_blocking=True)
                          for t in tuple(self))
            self._features[key] = moved
        return self._features[key]

    def rcr_features(self, config, device=None, dtype=torch.float32):
        device = torch.device('cpu' if device is None else device)
        key = ('rcr', config, str(device), dtype)
        if key not in self._features:
            env = self._env
            if (env.state.finish_time is not self._token
                    or len(env.solution.jo_idx) != self._step):
                raise RuntimeError('RCR features must be materialized before env.step/reset.')
            dynamic = self._rcr_dynamic
            kwargs = {} if dynamic is None else dict(
                action_mask=dynamic[2], machine_ready=dynamic[0], o_at=dynamic[1])
            self._features[key] = get_rcr(env, config).actor_features_torch(
                env, device=device, dtype=dtype, **kwargs)
        return self._features[key]


def config_dict(config):
    return asdict(config)
