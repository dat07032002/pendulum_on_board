"""TQC fine-tuning with fixed teacher replay and actor-behavior retention."""
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F

from sb3_contrib import TQC
from sb3_contrib.tqc.tqc import quantile_huber_loss
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update


class RetentionTQC(TQC):
    def configure_retention(
        self,
        teacher_data_path: str,
        actor_lr: float = 1e-6,
        critic_lr: float = 3e-5,
        actor_start_steps: int = 100_000,
        teacher_coef: float = 100.0,
        teacher_fraction: float = 0.5,
    ) -> None:
        data = np.load(teacher_data_path)
        required = ("observations", "actions", "next_observations", "rewards", "dones")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"teacher dataset missing arrays: {missing}")
        lengths = {key: len(data[key]) for key in required}
        if len(set(lengths.values())) != 1 or next(iter(lengths.values())) == 0:
            raise ValueError(f"invalid teacher dataset lengths: {lengths}")
        self.teacher_data = {key: np.asarray(data[key], dtype=np.float32) for key in required}
        self.actor_finetune_lr = float(actor_lr)
        self.critic_finetune_lr = float(critic_lr)
        self.actor_start_steps = int(actor_start_steps)
        self.teacher_coef = float(teacher_coef)
        self.teacher_fraction = float(teacher_fraction)
        if not 0.0 < self.teacher_fraction < 1.0:
            raise ValueError("teacher_fraction must be between zero and one")
        print(
            f"[retention] teacher transitions={lengths['observations']} "
            f"teacher_fraction={self.teacher_fraction:.2f} actor_start={self.actor_start_steps} "
            f"actor_lr={self.actor_finetune_lr:g} critic_lr={self.critic_finetune_lr:g} "
            f"teacher_coef={self.teacher_coef:g}",
            flush=True,
        )

    def _teacher_sample(self, batch_size: int) -> ReplayBufferSamples:
        indices = np.random.randint(0, len(self.teacher_data["observations"]), size=batch_size)

        def tensor(name: str) -> th.Tensor:
            return th.as_tensor(self.teacher_data[name][indices], device=self.device)

        return ReplayBufferSamples(
            observations=tensor("observations"),
            actions=tensor("actions"),
            next_observations=tensor("next_observations"),
            dones=tensor("dones"),
            rewards=tensor("rewards"),
            discounts=None,
        )

    @staticmethod
    def _join(a: ReplayBufferSamples, b: ReplayBufferSamples) -> ReplayBufferSamples:
        return ReplayBufferSamples(
            observations=th.cat((a.observations, b.observations)),
            actions=th.cat((a.actions, b.actions)),
            next_observations=th.cat((a.next_observations, b.next_observations)),
            dones=th.cat((a.dones, b.dones)),
            rewards=th.cat((a.rewards, b.rewards)),
            discounts=None,
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if not hasattr(self, "teacher_data"):
            raise RuntimeError("configure_retention() must be called before learn()")
        self.policy.set_training_mode(True)
        for group in self.actor.optimizer.param_groups:
            group["lr"] = self.actor_finetune_lr
        for group in self.critic.optimizer.param_groups:
            group["lr"] = self.critic_finetune_lr

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, teacher_losses = [], [], []
        actor_enabled = self.num_timesteps >= self.actor_start_steps

        for gradient_step in range(gradient_steps):
            teacher_batch_size = int(round(batch_size * self.teacher_fraction))
            online_batch_size = batch_size - teacher_batch_size
            online_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                online_batch_size, env=self._vec_normalize_env
            )
            teacher_data = self._teacher_sample(teacher_batch_size)
            replay_data = self._join(online_data, teacher_data)
            actual_batch_size = replay_data.observations.shape[0]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_quantiles = self.critic_target(replay_data.next_observations, next_actions)
                n_target_quantiles = (
                    self.critic.quantiles_total
                    - self.top_quantiles_to_drop_per_net * self.critic.n_critics
                )
                next_quantiles, _ = th.sort(next_quantiles.reshape(actual_batch_size, -1))
                next_quantiles = next_quantiles[:, :n_target_quantiles]
                target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                target_quantiles = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * target_quantiles
                )
                target_quantiles.unsqueeze_(dim=1)

            current_quantiles = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = quantile_huber_loss(
                current_quantiles, target_quantiles, sum_over_quantiles=False
            )
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if actor_enabled:
                qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(
                    dim=1, keepdim=True
                )
                rl_actor_loss = (ent_coef * log_prob - qf_pi).mean()
                student_teacher_actions = self.actor(teacher_data.observations, deterministic=True)
                teacher_loss = F.mse_loss(student_teacher_actions, teacher_data.actions)
                actor_loss = rl_actor_loss + self.teacher_coef * teacher_loss
                actor_losses.append(actor_loss.item())
                teacher_losses.append(teacher_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_frozen", float(not actor_enabled))
        self.logger.record("train/actor_loss", np.mean(actor_losses) if actor_losses else 0.0)
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/teacher_loss", np.mean(teacher_losses) if teacher_losses else 0.0)
        self.logger.record("train/actor_learning_rate", self.actor_finetune_lr)
        self.logger.record("train/critic_learning_rate", self.critic_finetune_lr)
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
