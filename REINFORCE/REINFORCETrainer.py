import os
import numpy as np
import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils import AverageMeter
from SchedulingModel import build_model_input, set_model_input_device
from RCR import RCRConfig, config_dict, get_rcr


# REINFORCE Trainer
class Trainer:
    def __init__(self, model, optimizer_params, training_params):
        self.rcr_config = getattr(model, 'rcr_config', RCRConfig(enabled=False))
        self.training_params = dict(training_params)
        self.requested_discount = self.training_params['discount_factor']
        if self.rcr_config.enabled:
            # The completion-loss preservation result is for the undiscounted
            # makespan objective. REINFORCE remains the optimizer
            self.training_params['discount_factor'] = 1.0

        self.last_rcr_stats = {}
        self._rcr_logged = False
        self.optimizer_params = optimizer_params

        # RESCHED's Runner does not pass its device into the Trainer. For the
        # RCR build, use CUDA automatically when available; RCR_DEVICE=cpu can
        # explicitly force the original CPU execution path.
        current_device = next(model.parameters()).device
        requested_device = os.environ.get('RCR_DEVICE', '').strip()
        if requested_device:
            self.device = torch.device(requested_device)
        elif current_device.type != 'cpu':
            self.device = current_device
        elif self.rcr_config.enabled and torch.cuda.is_available():
            self.device = torch.device('cuda', torch.cuda.current_device())
        else:
            self.device = current_device

        self.model = model.to(self.device)
        set_model_input_device(self.device)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.logger = lambda x: print(x)

    def train(self, env, first_epoch=False):
        score_am, loss_am = AverageMeter(), AverageMeter()
        cfg = self.rcr_config

        if cfg.enabled and not self._rcr_logged:
            self.logger('RCR-REINFORCE config: {} | discount {} -> 1.0 | device={}'.format(
                config_dict(cfg), self.requested_discount, self.device))
            self._rcr_logged = True

        totals = dict(certificate_count=0, horizon_sum=0.,
                      preprocess_seconds=0., feature_seconds=0., reward_seconds=0.)
        base_reward_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        auxiliary_reward_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        delta_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        positive_certificates = torch.zeros((), dtype=torch.long, device=self.device)

        self.model.train()
        self.model.set_decode_type('sampling')
        loop_cnt = 0

        # RCR Actor features are produced in no-grad mode. Keep the exact
        # feature snapshot used during sampling and reuse it for gradient replay.
        feature_dtype = (
            self.model.duration_embedding.weight.dtype
            if hasattr(self.model, 'duration_embedding')
            else next(self.model.parameters()).dtype
        )

        episode = 0
        while episode < self.training_params['episode']:
            remaining = self.training_params['episode'] - episode
            bs = min(self.training_params['batch_size'], remaining)
            env.generate_data(bs)

            state = env.reset_state()
            rcr = get_rcr(env, cfg) if (cfg.use_actor or cfg.use_reward) else None
            if rcr is not None:
                totals['horizon_sum'] += rcr.witness.makespan.sum()

            reward_list = []
            replay_states = []
            replay_actions = []
            replay_rcr_features = []
            episode_scaler = None
            done = False

            # ------------------------------------------------------------
            # PASS 1: rollout without autograd.
            # ------------------------------------------------------------
            while not done:
                previous_estimate = state.est_last_makespan if cfg.enabled else None
                model_input, scaler = build_model_input(
                    state, env, device=self.device)

                with torch.no_grad():
                    action, prob = self.model(model_input)

                    if cfg.use_actor:
                        rcr_feature = model_input.rcr_features(
                            cfg, device=self.device, dtype=feature_dtype)
                        replay_rcr_features.append(
                            rcr_feature.detach().cpu())
                    else:
                        replay_rcr_features.append(None)

                    # Move only replay data to CPU. No forward activations or
                    # autograd graph survive this iteration.
                    replay_states.append(
                        tuple(t.detach().cpu() for t in tuple(model_input)))
                    replay_actions.append(
                        tuple(a.detach().cpu() for a in action))

                state, reward, done = env.step(action)

                if cfg.enabled:
                    base_np = previous_estimate - state.est_last_makespan
                    base_reward = torch.as_tensor(
                        base_np, dtype=prob.dtype, device=self.device)
                    reward = base_reward
                    base_reward_sum += base_reward.detach().sum()

                    if cfg.use_reward:
                        reward = rcr.compose_reward_torch(
                            env, base_reward,
                            device=self.device, dtype=prob.dtype)
                        auxiliary_reward_sum += rcr.last_aux_reward.sum()
                        delta_sum += rcr.last_delta.sum()
                        positive_certificates += (rcr.last_delta > 0).sum()
                        totals['certificate_count'] += bs
                elif not isinstance(reward, torch.Tensor):
                    reward = torch.as_tensor(
                        reward, dtype=prob.dtype, device=self.device)

                if episode_scaler is None:
                    if not isinstance(scaler, torch.Tensor):
                        episode_scaler = torch.as_tensor(
                            scaler, dtype=prob.dtype, device=self.device)
                    else:
                        episode_scaler = scaler.to(
                            device=self.device,
                            dtype=prob.dtype,
                            non_blocking=True)

                # Rewards carry no policy graph in REINFORCE.
                reward_list.append(reward.detach())

            rewards = torch.stack(reward_list, dim=0)
            returns = self.get_return_tensor(rewards)
            returns = returns / episode_scaler.reshape(1, bs)
            advantage = (returns - returns.mean(dim=1, keepdim=True)).detach()

            # ------------------------------------------------------------
            # PASS 2: replay the SAME sampled actions with the SAME policy
            # ------------------------------------------------------------
            self.model.set_decode_type('teacher_forcing')
            self.optimizer.zero_grad(set_to_none=True)
            loss_total = torch.zeros(
                (), dtype=returns.dtype, device=self.device)

            for step, (snapshot, action_cpu, feature_cpu) in enumerate(
                    zip(replay_states, replay_actions, replay_rcr_features)):
                seq = action_cpu[0].to(self.device, non_blocking=True)
                machine = action_cpu[1].to(self.device, non_blocking=True)

                feature_override = None
                if feature_cpu is not None:
                    feature_override = feature_cpu.to(
                        device=self.device,
                        dtype=feature_dtype,
                        non_blocking=True)

                _, selected_prob = self.model(
                    snapshot,
                    action=(seq, machine),
                    rcr_features_override=feature_override)

                loss_step = -(
                    advantage[step] * selected_prob.log()
                ).mean()

                loss_step.backward()
                loss_total += loss_step.detach()

                del selected_prob, loss_step, feature_override, seq, machine

            self.optimizer.step()
            self.model.set_decode_type('sampling')

            loss_value = loss_total.item()
            loss_am.update(loss_value, bs)

            score_mean = state.finish_time.max(axis=1).max(axis=1).mean()
            score_am.update(score_mean.item(), bs)

            if rcr is not None:
                totals['preprocess_seconds'] += rcr.preprocess_seconds
                totals['feature_seconds'] += rcr.feature_seconds
                totals['reward_seconds'] += rcr.reward_seconds

            episode += bs

            if first_epoch and (loop_cnt <= 10):
                self.logger(
                    'Epoch 1: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                    .format(
                        episode,
                        self.training_params['episode'],
                        100. * episode / self.training_params['episode'],
                        score_am.avg,
                        loss_am.avg))
            loop_cnt += 1

        self.scheduler.step()

        if cfg.enabled:
            count = max(totals['certificate_count'], 1)
            self.last_rcr_stats = dict(
                totals,
                mean_base_return=(base_reward_sum / max(episode, 1)).item(),
                mean_auxiliary_return=(auxiliary_reward_sum / max(episode, 1)).item(),
                mean_delta=(delta_sum / count).item(),
                positive_fraction=(positive_certificates.float() / count).item(),
                mean_horizon=totals['horizon_sum'] / max(episode, 1),
            )
            self.logger(
                'RCR-REINFORCE: base={:.4f}, aux={:.4f}, Delta={:.4f}, '
                'positive={:.2%}, H={:.4f}, setup/actor/reward={:.3f}/{:.3f}/{:.3f}s'
                .format(
                    self.last_rcr_stats['mean_base_return'],
                    self.last_rcr_stats['mean_auxiliary_return'],
                    self.last_rcr_stats['mean_delta'],
                    self.last_rcr_stats['positive_fraction'],
                    self.last_rcr_stats['mean_horizon'],
                    totals['preprocess_seconds'],
                    totals['feature_seconds'],
                    totals['reward_seconds'],
                )
            )

        return score_am.avg, loss_am.avg, self.model.state_dict()

    def get_return_tensor(self, rewards):
        """REINFORCE reward-to-go entirely on the model device."""
        discount_factor = self.training_params['discount_factor']
        if discount_factor == 1.0:
            return torch.flip(torch.cumsum(torch.flip(rewards, dims=(0,)), dim=0), dims=(0,))
        returns = torch.empty_like(rewards)
        g = torch.zeros_like(rewards[0])
        for t in range(rewards.shape[0] - 1, -1, -1):
            g = g * discount_factor + rewards[t]
            returns[t] = g
        return returns

    def get_return(self, reward_list):
        discount_factor = self.training_params['discount_factor']
        return_list = []
        g = 0
        for reward in reversed(reward_list):
            g = g * discount_factor + reward
            return_list.insert(0, g)

        return return_list

    def set_logger(self, logger):
        self.logger = logger

    def load_checkpoint(self, optimizer_params, scheduler_params, path):
        try:
            self.optimizer.load_state_dict(optimizer_params)
            self.logger('Optimizer loaded from {}'.format(path))
        except KeyError:
            self.logger('Saved Optimizer can not be used and new Optimizer is used instead')

        try:
            self.scheduler.load_state_dict(scheduler_params)
            self.logger('Scheduler loaded from {}'.format(path))
        except KeyError:
            self.logger('Saved Scheduler can not be used and new Scheduler is used instead')
