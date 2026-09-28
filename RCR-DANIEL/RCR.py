"""
RCR-DANIEL: conditioned Actor features + completion-loss penalty.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter
import json
import os

import numpy as np


@dataclass(frozen=True)
class RCRConfig:
    # Retain the prior RCR horizon/group/window/temperature defaults.
    enabled: bool = True
    actor_features: bool = True
    reward_shaping: bool = True
    capacity_mode: str = 'combined'  # 'combined', 'tw', or 'global'
    extra_group_count: int = 8
    window_bins: int = 8
    pressure_temperature: float = 0.1
    reward_weight: float = 0.3
    candidate_chunk_size: int = 32
    certificate_epsilon: float = 1e-7
    feature_pooling: str = 'balanced'  # 'legacy' reproduces the old pooling.
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
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
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
    
    if options is None:
        from params import configs
        options = getattr(configs, 'rcr', None)
    if options is None:
        options = json.loads(os.environ.get('RCR_CONFIG', '{}'))
    return RCRConfig.from_options(options)


@dataclass
class AnalyticState:
    remaining: np.ndarray       # [B,N], unscheduled REAL operations only
    job_ready: np.ndarray       # [B,J], retain actual ends of completed jobs
    machine_ready: np.ndarray   # [B,M], ends of assigned processing
    time: np.ndarray            # [B], next native decision time


@dataclass
class Witness:
    actions: np.ndarray         # [B,Nmax], job*M+machine; padding=-1
    operations: np.ndarray      # [B,Nmax], original operation ids; padding=-1
    machines: np.ndarray
    starts: np.ndarray          # Native normalized time units
    ends: np.ndarray
    makespan: np.ndarray


class SchedulingAdapter:
    """
    Exact transition for both supplied DANIEL environment classes.
    """

    def __init__(self, env):
        self.duration = np.asarray(env.unmasked_op_pt, dtype=np.float64).copy()
        self.eligible = np.asarray(env.process_relation, dtype=bool).copy()
        self.batch_size, self.n, self.m = self.duration.shape
        self.j = env.number_of_jobs
        self.counts = env.job_length.sum(-1).astype(np.int64)
        self.op_valid = np.arange(self.n)[None, :] < self.counts[:, None]
        expected = getattr(env, 'env_number_of_ops', np.full(self.batch_size, self.n))
        if ((env.job_length <= 0).any() or not np.array_equal(self.counts, expected)
                or not np.isfinite(self.duration).all() or (self.duration < 0).any()
                or not self.eligible.any(-1)[self.op_valid].all()):
            raise ValueError('RCR requires valid job chains and at least one eligible '
                             'machine per real operation AFTER native normalization. '
                             'The uploaded normalization is intentionally unchanged.')
        if not np.array_equal(self.eligible, self.duration > 0):
            raise ValueError('Native processing times and eligibility disagree.')
        self.pmin = np.where(self.op_valid,
                             np.where(self.eligible, self.duration, np.inf).min(-1), 0.)
        self.op_job = np.zeros((self.batch_size, self.n), dtype=np.int64)
        self.job_valid = np.ones((self.batch_size, self.j), dtype=bool)
        self.job_member = np.zeros((self.batch_size, self.j, self.n), dtype=np.float64)
        self.prefix = np.zeros_like(self.pmin)
        self.suffix = np.zeros_like(self.pmin)
        self.first = np.array(env.job_first_op_id, copy=True)
        self.last = np.array(env.job_last_op_id, copy=True)
        for b in range(self.batch_size):
            for j in range(self.j):
                ops = np.arange(self.first[b, j], self.last[b, j] + 1)
                self.op_job[b, ops] = j
                self.job_member[b, j, ops] = 1.
                work = self.pmin[b, ops]
                self.prefix[b, ops] = np.cumsum(work) - work
                self.suffix[b, ops] = work.sum() - np.cumsum(work)
        self.total_work = np.einsum('bjn,bn->bj', self.job_member, self.pmin)

    def export_state(self, env):
        remaining = self.op_valid & ~np.asarray(env.op_scheduled_flag, dtype=bool)
        # numpy.ma exposes an arbitrary sentinel when all pairs are masked.
        # A terminal state has no next event; canonically use its makespan.
        time = np.where(remaining.any(-1), env.next_schedule_time,
                        np.max(env.candidate_free_time, axis=1))
        return AnalyticState(remaining.copy(), env.candidate_free_time.copy(),
                             env.mch_free_time.copy(), time.copy())

    def candidates(self, remaining, ids):
        left = np.einsum('qjn,qn->qj', self.job_member[ids], remaining).astype(np.int64)
        return np.minimum(self.last[ids], self.last[ids] + 1 - left), left > 0

    def action_mask(self, env):
        # Native reset restores the state tensor and candidate relation, but
        # leaves the previous episode's raw dynamic_pair_mask array in place.
        
        pair_time = np.maximum(env.candidate_free_time[:, :, None],
                               env.mch_free_time[:, None, :])
        return (~env.candidate_process_relation &
                (pair_time <= env.next_schedule_time[:, None, None]))

    def preview(self, state, batch_ids, operations, machines):
        ids = np.asarray(batch_ids, dtype=np.int64)
        rows = np.arange(len(ids))
        result = AnalyticState(state.remaining[ids].copy(), state.job_ready[ids].copy(),
                               state.machine_ready[ids].copy(), state.time[ids].copy())
        jobs = self.op_job[ids, operations]
        ends = np.maximum(result.job_ready[rows, jobs], result.machine_ready[rows, machines])
        ends = ends + self.duration[ids, operations, machines]
        result.job_ready[rows, jobs] = ends
        result.machine_ready[rows, machines] = ends
        result.remaining[rows, operations] = False
        candidate, live = self.candidates(result.remaining, ids)
        eligible = self.eligible[ids[:, None], candidate] & live[:, :, None]
        pair_time = np.maximum(result.job_ready[:, :, None], result.machine_ready[:, None, :])
        next_time = np.where(eligible, pair_time, np.inf).min((1, 2))
        result.time = np.where(live.any(-1), next_time, result.job_ready.max(-1))
        return result

    def make_witness(self, env):
        """
        Two deterministic legal native rollouts, without changing env or RNG.
        SPT / MWKR tie breaks: earliest completion, job id, then machine id.
        The complete better witness is retained per instance.
        """
        witnesses = []
        for rule in ('spt', 'mwkr'):
            reference = deepcopy(env)
            reference.rcr_config = RCRConfig(enabled=False)
            reference.rcr = None
            reference.reset()
            arrays = {name: np.full((self.batch_size, self.n), -1, dtype=np.int64)
                      for name in ('actions', 'operations', 'machines')}
            arrays.update({name: np.zeros((self.batch_size, self.n), dtype=np.float64)
                           for name in ('starts', 'ends')})
            done = np.zeros(self.batch_size, dtype=bool)
            for t in range(self.n):
                ids = np.flatnonzero(~done)
                action = []
                remaining_work = np.einsum('bjn,bn->bj', self.job_member,
                    self.pmin * self.export_state(reference).remaining)
                legal_pairs = self.action_mask(reference)
                for b in ids:
                    jobs, machines = np.nonzero(legal_pairs[b])
                    if not len(jobs):
                        raise ValueError('Native environment has no legal reference action.')
                    ops = reference.candidate[b, jobs]
                    proc = self.duration[b, ops, machines]
                    starts = np.maximum(reference.candidate_free_time[b, jobs],
                                        reference.mch_free_time[b, machines])
                    primary = proc if rule == 'spt' else -remaining_work[b, jobs]
                    k = np.lexsort((machines, jobs, starts + proc, primary))[0]
                    chosen = int(jobs[k] * self.m + machines[k])
                    action.append(chosen)
                    arrays['actions'][b, t] = chosen
                    arrays['operations'][b, t] = ops[k]
                    arrays['machines'][b, t] = machines[k]
                    arrays['starts'][b, t] = starts[k]
                _, _, done = reference.step(np.asarray(action, dtype=np.int64))
                done = np.asarray(done, dtype=bool)
                arrays['ends'][ids, t] = reference.op_ct[ids, arrays['operations'][ids, t]]
                if np.asarray(done, dtype=bool).all():
                    break
            witness = Witness(**arrays, makespan=reference.op_ct.max(-1))
            self.validate_witness(witness)
            witnesses.append(witness)
        first, second = witnesses
        choose = first.makespan <= second.makespan
        return Witness(**{name: np.where(choose if name == 'makespan' else choose[:, None],
                                          getattr(first, name), getattr(second, name))
                          for name in Witness.__dataclass_fields__})

    def validate_witness(self, witness):
        for b, count in enumerate(self.counts):
            ops, mas = witness.operations[b, :count], witness.machines[b, :count]
            starts, ends = witness.starts[b, :count], witness.ends[b, :count]
            tol = 1e-10 * max(1., witness.makespan[b])
            if not np.array_equal(np.sort(ops), np.arange(count)):
                raise ValueError('Reference must assign every real operation exactly once.')
            if ((starts < -tol).any() or not self.eligible[b, ops, mas].all()
                    or not np.allclose(ends - starts, self.duration[b, ops, mas],
                                       rtol=1e-10, atol=tol)):
                raise ValueError('Invalid reference processing times/eligibility.')
            order = np.argsort(ops)
            for j in range(self.j):
                job_ops = np.arange(self.first[b, j], self.last[b, j] + 1)
                if (starts[order[job_ops[1:]]] + tol < ends[order[job_ops[:-1]]]).any():
                    raise ValueError('Reference violates job precedence.')
            for m in range(self.m):
                which = np.flatnonzero(mas == m)
                which = which[np.argsort(starts[which], kind='stable')]
                if (starts[which[1:]] + tol < ends[which[:-1]]).any():
                    raise ValueError('Reference has overlapping machine use.')
            if abs(ends.max() - witness.makespan[b]) > tol:
                raise ValueError('Reference makespan mismatch.')


class RCRCore:

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
                          & self.group_valid[:, :, None]
                          & a.op_valid[:, None, :]).astype(np.float64)
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

    @staticmethod
    def mandatory_work(pmin, earliest, latest, left, right):
        return np.maximum(0., np.minimum(np.minimum(pmin, right - left),
                          np.minimum(earliest + pmin - left, right - latest + pmin)))

    def constraints(self, state, batch_ids=None):
        
        a = self.adapter
        ids = np.arange(a.batch_size) if batch_ids is None else np.asarray(batch_ids)
        q = len(ids)
        remaining = state.remaining
        pmin = a.pmin[ids]
        work = np.einsum('qjn,qn->qj', a.job_member[ids], remaining * pmin)
        completed_work = a.total_work[ids] - work
        jobs = a.op_job[ids]
        rows = np.arange(q)[:, None]
        job_ready = np.where(work > 0,
                             np.maximum(state.job_ready, state.time[:, None]),
                             state.job_ready)
        earliest = (job_ready[rows, jobs] + a.prefix[ids]
                    - completed_work[rows, jobs])
        latest = self.horizon[ids, None] - a.suffix[ids]
        left, right = self.windows[ids, :, 0], self.windows[ids, :, 1]
        mandatory = self.mandatory_work(pmin[:, :, None], earliest[:, :, None],
                                       latest[:, :, None], left[:, None, :], right[:, None, :])
        mandatory *= remaining[:, :, None]
        demand = np.matmul(self.op_member[ids], mandatory)
        free = np.maximum(0., right[:, None, :] -
                          np.maximum(left[:, None, :],
                                     np.maximum(state.machine_ready,
                                                state.time[:, None])[:, :, None]))
        capacity = np.matmul(self.machine_member[ids], free)
        violation = demand - capacity
        h = self.horizon[ids, None]
        job_x = (job_ready + work - h) / h
        global_x = violation[:, :, 0] / (h * self.group_size[ids])
        n_group = np.einsum('qgn,qn->qg', self.op_member[ids], remaining).clip(min=1)
        local_x = violation[:, :, 1:] / (h[:, :, None] * n_group[:, :, None])
        x = np.concatenate((job_x, global_x, local_x.reshape(q, -1)), axis=1)
        x = np.where(self.valid[ids], x, 0.)
        largest = np.where(self.valid[ids], x, -np.inf).max(-1)
        delta = self.horizon[ids] * np.maximum(0., largest - self.config.certificate_epsilon)
        return x, delta

    @property
    def families(self):
        return (slice(0, self.job_end), slice(self.job_end, self.global_end),
                slice(self.global_end, self.valid.shape[1]))

    def active_constraints(self, state):
        """
        An Actor pooling mask, NEVER a hard action or feasibility mask.
        """
        a = self.adapter
        job_live = np.einsum('bjn,bn->bj', a.job_member, state.remaining) > 0
        group_live = ((np.einsum('bgn,bn->bg', self.op_member, state.remaining) > 0)
                      & self.group_valid)
        window_live = self.windows[:, 1:, 1] > state.time[:, None]
        local_live = group_live[:, :, None] & window_live[:, None, :]
        return self.valid & np.concatenate((job_live, group_live,
                                            local_live.reshape(a.batch_size, -1)), axis=1)

    def log_weights(self, x, batch_ids, active=None):
        valid = self.valid[batch_ids] if active is None else active & self.valid[batch_ids]
        if self.config.feature_pooling == 'legacy':
            z = np.where(valid, x / self.config.pressure_temperature, -np.inf)
            maximum = z.max(-1, keepdims=True)
            logsum = maximum + np.log(np.exp(z - maximum).sum(-1, keepdims=True))
            logw = np.where(valid, z - logsum, 0.)
            return logw, np.where(valid, np.exp(logw), 0.)

        # Equal mass per nonempty FAMILY, independent of its constraint count.
        # Within a family retain temperature-weighted bottleneck sensitivity.
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
            logw[:, part] = np.where(mask, z - logsum - np.log(family_count[:, None]), 0.)
            w[:, part] = np.where(mask, np.exp(logw[:, part]), 0.)
        return logw, w

    def raw_actor_features(self, state, action_mask, candidate):
        a, cfg = self.adapter, self.config
        x, _ = self.constraints(state)
        active = (self.active_constraints(state) if cfg.feature_pooling == 'balanced'
                  else self.valid)
        logw, weights = self.log_weights(x, np.arange(a.batch_size), active)
        batch_ids, sequences, machines = np.nonzero(action_mask)
        features = np.zeros(action_mask.shape + (4,), dtype=np.float64)
        for start in range(0, len(batch_ids), cfg.candidate_chunk_size):
            sl = slice(start, start + cfg.candidate_chunk_size)
            ids, seq, mas = batch_ids[sl], sequences[sl], machines[sl]
            post = a.preview(state, ids, candidate[ids, seq], mas)
            post_x, _ = self.constraints(post, ids)
            # A common parent-state mask is essential to the finite KL and
            # the four-term local log-sum-exp identity. Do not remask post.
            post_logw, _ = self.log_weights(post_x, ids, active[ids])
            change = weights[ids] * (post_x - x[ids])
            features[ids, seq, mas, 0] = change[:, :self.job_end].sum(-1)
            features[ids, seq, mas, 1] = change[:, self.job_end:self.global_end].sum(-1)
            features[ids, seq, mas, 2] = change[:, self.global_end:].sum(-1)
            kl = (weights[ids] * (logw[ids] - post_logw)).sum(-1)
            features[ids, seq, mas, 3] = cfg.pressure_temperature * np.maximum(kl, 0.)
        return features

    def actor_features(self, state, action_mask, candidate):
        raw = self.raw_actor_features(state, action_mask, candidate)
        return condition_actor_features(raw, action_mask, self.config)


def condition_actor_features(raw, action_mask, config):
    """
    Per-state legal-action conditioning; independent of batch and PPO replay.
    """
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
    return np.clip(centered / scale, -config.actor_feature_clip, config.actor_feature_clip)


class RCREpisode:
    def __init__(self, env):
        start = perf_counter()
        self.config = env.rcr_config
        self.adapter = SchedulingAdapter(env)
        self.witness = self.adapter.make_witness(env)
        self.core = RCRCore(self.adapter, self.witness.makespan, self.config)
        self.preprocess_seconds = perf_counter() - start
        self.reset_statistics()

    def reset_statistics(self):
        self.feature_seconds = 0.
        self.reward_seconds = 0.
        b = self.adapter.batch_size
        self.last_delta = np.zeros(b, dtype=np.float64)
        self.last_base_reward = np.zeros(b, dtype=np.float64)
        self.last_aux_reward = np.zeros(b, dtype=np.float64)
        self.last_total_reward = np.zeros(b, dtype=np.float64)

    def actor_features(self, env):
        start = perf_counter()
        result = self.core.actor_features(self.adapter.export_state(env),
                                         self.adapter.action_mask(env), env.candidate)
        self.feature_seconds += perf_counter() - start
        return result

    def compose_reward(self, env, base_reward, active_ids=None):
        """
        Add the completion-loss penalty to DANIEL's native reward.
        """
        start = perf_counter()
        base = np.asarray(base_reward, dtype=np.float64)
        if base.shape != (self.adapter.batch_size,):
            raise ValueError('base_reward must have shape [batch].')

        _, delta = self.core.constraints(self.adapter.export_state(env))
        counts = self.adapter.counts.astype(np.float64)
        aux = -self.config.reward_weight * delta / counts

        if active_ids is not None:
            active = np.zeros(self.adapter.batch_size, dtype=bool)
            active[np.asarray(active_ids, dtype=np.int64)] = True
            aux = np.where(active, aux, 0.)

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

    def statistics(self):
        return dict(config=asdict(self.config),
                    horizon_native=self.witness.makespan.tolist(),
                    preprocess_seconds=self.preprocess_seconds,
                    feature_seconds=self.feature_seconds,
                    reward_seconds=self.reward_seconds,
                    last_delta=self.last_delta.tolist(),
                    last_base_reward=self.last_base_reward.tolist(),
                    last_aux_reward=self.last_aux_reward.tolist(),
                    last_total_reward=self.last_total_reward.tolist())


def attach_actor_features(env):
    if env.rcr is None or not env.rcr_config.use_actor:
        return
    # EnvState.update/deepcopy creates a fresh eight-column tensor first.
    # Concatenation creates an immutable-by-convention twelve-column snapshot.
    import torch
    pairs = env.state.fea_pairs_tensor
    if pairs.shape[-1] != 8:
        raise RuntimeError('RCR features must be attached once to fresh native pairs.')
    extra = torch.as_tensor(env.rcr.actor_features(env), dtype=pairs.dtype, device=pairs.device)
    env.state.fea_pairs_tensor = torch.cat((pairs, extra), dim=-1)


def reset_rcr(env, new_problem=False):
    # Reward-only ablations still need the episode-local certificate cache even
    # when no four-dimensional features are appended to the Actor.
    if not (env.rcr_config.use_actor or env.rcr_config.use_reward):
        env.rcr = None
        return
    if new_problem or env.rcr is None or env.rcr.config != env.rcr_config:
        # Clear an old episode before cloning the native environment.
        env.rcr = None
        env.rcr = RCREpisode(env)
    else:
        env.rcr.reset_statistics()
    attach_actor_features(env)


def config_dict(config):
    return asdict(config)
