# Add reference to paper

import warnings

import escnn
import torch
import torch.nn as nn
import torch.optim as optim
from escnn.nn import FieldType
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.algorithms import PPO
from morphosymm_rl.symm_utils import configure_observation_space_representations


class PPOSymmDataAugmented:
    """Proximal Policy Optimization algorithm with data augmentation via symmetries."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,  # added
        **morphologycal_symmetries_cfg,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # For now, we ignore this
        self.rnd = None
        self.rnd_optimizer = None
        self.symmetry = None

        # MorphoSymm components
        obs_space_names = morphologycal_symmetries_cfg["obs_space_names"]
        action_space_names = morphologycal_symmetries_cfg["action_space_names"]
        joints_order = morphologycal_symmetries_cfg["joints_order"]
        history_length = morphologycal_symmetries_cfg["history_length"]
        robot_name = morphologycal_symmetries_cfg["robot_name"]

        G, obs_reps = configure_observation_space_representations(robot_name, obs_space_names, joints_order)

        obs_space_reps = [obs_reps[n] for n in obs_space_names] * history_length
        act_space_reps = [obs_reps[n] for n in action_space_names]
        # rep_extra_obs = [rep_R3, rep_R3_pseudo, trivial_rep, trivial_rep, rep_friction, rep_R3, trivial_rep, trivial_rep, rep_kin_three, rep_kin_three, rep_kin_three, rep_kin_three, trivial_rep, trivial_rep]

        gspace = escnn.gspaces.no_base_space(G)
        self.in_field_type = FieldType(gspace, obs_space_reps)
        self.out_field_type = FieldType(gspace, act_space_reps)

        self.critic_in_field_type = FieldType(gspace, obs_space_reps)
        self.num_replica = len(G.elements)
        self.G = G

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs * self.num_replica,  # added here,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def test_mode(self):
        self.policy.test()

    def train_mode(self):
        self.policy.train()

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Augment the transition
        self.augment_transitions()

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def augment_transitions(self):
        t = self.transition
        out_field_type = self.out_field_type
        in_field_type = self.in_field_type
        critic_in_field_type = self.critic_in_field_type
        G = self.G
        t.actions = torch.cat(
            [t.actions] + [out_field_type.transform_fibers(t.actions, g) for g in G.elements[1:]],
            dim=0,
        )
        t.actions_log_prob = torch.cat([t.actions_log_prob] * self.num_replica, dim=0)
        t.action_mean = torch.cat(
            [t.action_mean] + [out_field_type.transform_fibers(t.action_mean, g) for g in G.elements[1:]],
            dim=0,
        )
        t.action_sigma = torch.abs(
            torch.cat(
                [t.action_sigma] + [out_field_type.transform_fibers(t.action_sigma, g) for g in G.elements[1:]],
                dim=0,
            )
        )
        t.values = torch.cat([t.values] * self.num_replica, dim=0)
        t.rewards = torch.cat([t.rewards] * self.num_replica, dim=0)
        t.dones = torch.cat([t.dones] * self.num_replica, dim=0)
        t.observations = torch.cat(
            [t.observations] + [in_field_type.transform_fibers(t.observations, g) for g in G.elements[1:]],
            dim=0,
        )
        t.privileged_observations = torch.cat(
            [t.privileged_observations]
            + [critic_in_field_type.transform_fibers(t.privileged_observations, g) for g in G.elements[1:]],
            dim=0,
        )

    def augment_values(self, values):
        values = torch.cat([values] * self.num_replica, dim=0)
        return values

    def compute_returns(self, last_critic_obs):
        last_values = self.policy.evaluate(last_critic_obs).detach()
        last_values = self.augment_values(last_values)
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = None
        mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:
            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch)
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding.detach())

            # Gradient step
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()

            # Store the losses
            # Check that no loss is NaN
            assert not torch.isnan(value_loss), "Loss is NaN"
            assert not torch.isnan(surrogate_loss), "Surrogate loss is NaN"
            assert not torch.isnan(entropy_batch.mean()), "Entropy loss is NaN"

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates

        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

        # return mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, None
