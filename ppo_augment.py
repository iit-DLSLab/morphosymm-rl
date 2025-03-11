# Add reference to paper

import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

#from morpho_symm.nn.test_EMLP import get_kinematic_three_rep_two, get_ground_reaction_forces_rep_two, get_friction_rep
from test_EMLP import get_kinematic_three_rep_two, get_ground_reaction_forces_rep_two, get_friction_rep


import escnn
from escnn.nn import FieldType
from hydra import compose, initialize

from morpho_symm.utils.robot_utils import load_symmetric_system

from hydra.core.global_hydra import GlobalHydra

class PPOAugmented:
    """Proximal Policy Optimization algorithm with data augmentation via symmetries."""

    actor_critic: ActorCritic
    """The actor critic module."""

    def __init__(self,
                 actor_critic,
                 #task, #to remove
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
                 device='cpu',
                 normalize_advantage_per_mini_batch=False, #added
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
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

        # MorphoSymm components - our addition goes here!!
        global G
        GlobalHydra.instance().clear()
        initialize(config_path="./MorphoSymm/morpho_symm/cfg/robot", version_base='1.3')
        robot_name = 'a1'  # or any of the robots in the library (see `/morpho_symm/cfg/robot`)
        robot_cfg = compose(config_name=f"{robot_name}.yaml")
        robot, G = load_symmetric_system(robot_cfg=robot_cfg)

        # We use ESCNN to handle the group/representation-theoretic concepts and for the construction of equivariant neural networks.
        gspace = escnn.gspaces.no_base_space(G)
        # Get the relevant group representations.
        rep_QJ = G.representations["Q_js"]  # Used to transform joint-space position coordinates q_js ∈ Q_js
        rep_TqQJ = G.representations["TqQ_js"]  # Used to transform joint-space velocity coordinates v_js ∈ TqQ_js
        # IT WAS RD and RD_pseudo
        rep_O3 = G.representations["R3"]  # Used to transform the linear momentum l ∈ R3
        rep_O3_pseudo = G.representations["R3_pseudo"]  # Used to transform the angular momentum k ∈ R3
        trivial_rep = G.trivial_representation
        rep_kin_three = get_kinematic_three_rep_two(G)
        rep_friction = get_friction_rep(G, rep_kin_three)
        # Define the input and output FieldTypes using the representations of each geometric object.
        # Representation of x := [q, v] ∈ Q_js x TqQ_js      =>    ρ_X_js(g) := ρ_Q_js(g) ⊕ ρ_TqQ_js(g)  | g ∈ G
        """obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    obs_commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                    clock_data,
                )
                if tensor is not None
            ],
            dim=-1,
        )"""
        base_transition = [rep_O3, rep_O3_pseudo, rep_O3, rep_O3_pseudo, rep_QJ, rep_TqQJ, rep_QJ, rep_kin_three, rep_kin_three] * 3
        rep_extra_obs = [rep_O3, rep_O3_pseudo, trivial_rep, trivial_rep, rep_friction, rep_O3, trivial_rep, trivial_rep, rep_kin_three, rep_kin_three, rep_kin_three, rep_kin_three, trivial_rep, trivial_rep]


        self.in_field_type = FieldType(gspace, base_transition)
        # Representation of y := [l, k] ∈ R3 x R3            =>    ρ_Y_js(g) := ρ_O3(g) ⊕ ρ_O3pseudo(g)  | g ∈ G
        self.out_field_type = FieldType(gspace, [rep_QJ])

        #self.critic_in_field_type = FieldType(gspace, base_transition + rep_extra_obs)
        self.critic_in_field_type = FieldType(gspace, base_transition)

        self.num_replica = len(G.elements)
        self.G = G

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            num_envs * self.num_replica, #added here,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            rnd_state_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Augment the transition
        self.augment_transitions()

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def augment_transitions(self):
        t = self.transition
        out_field_type = self.out_field_type
        in_field_type = self.in_field_type
        critic_in_field_type = self.critic_in_field_type
        G = self.G
        t.actions = torch.cat([t.actions] + [out_field_type.transform_fibers(t.actions, g) for g in G.elements[1:]], dim=0)
        t.actions_log_prob = torch.cat([t.actions_log_prob] * self.num_replica, dim=0)
        t.action_mean = torch.cat([t.action_mean] + [out_field_type.transform_fibers(t.action_mean, g) for g in G.elements[1:]], dim=0)
        t.action_sigma = torch.abs(torch.cat([t.action_sigma] + [out_field_type.transform_fibers(t.action_sigma, g) for g in G.elements[1:]], dim=0))
        t.values = torch.cat([t.values] * self.num_replica, dim=0)
        t.rewards = torch.cat([t.rewards] * self.num_replica, dim=0)
        t.dones = torch.cat([t.dones] * self.num_replica, dim=0)
        t.observations = torch.cat([t.observations] + [in_field_type.transform_fibers(t.observations, g) for g in G.elements[1:]], dim=0)
        t.critic_observations = torch.cat([t.critic_observations] + [critic_in_field_type.transform_fibers(t.critic_observations, g) for g in G.elements[1:]], dim=0)


    def augment_values(self, values):
        values = torch.cat([values] * self.num_replica, dim=0)
        return values


    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        last_values = self.augment_values(last_values)
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )


    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = None
        mean_symmetry_loss = None

        # generator for mini batches
        if self.actor_critic.is_recurrent:
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
            # Note: we need to do this because we updated the actor_critic with the new parameters
            # -- actor
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            entropy_batch = self.actor_critic.entropy[:original_batch_size]

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
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()

            # Store the losses
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

        return mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, None
