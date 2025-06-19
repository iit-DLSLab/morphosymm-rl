# Add reference to paper

import escnn
import numpy as np
import torch
import torch.nn as nn
from escnn.nn import EquivariantModule, FieldType, GeometricTensor
from torch.distributions import Normal

from morphosymm_rl.symm_utils import configure_observation_space_representations
from rsl_rl.rsl_rl.modules.actor_critic import ActorCritic

G = None


class ActorCriticSymmEquivariantNN(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **morphologycal_symmetries_cfg,
    ):
        # Instead of calling ActorCritic.__init__, call torch.nn.Module.__init__
        torch.nn.Module.__init__(self)
        # Cache init args for export function
        self._ac_kwargs = dict(
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )

        # MorphoSymm components
        obs_space_names = morphologycal_symmetries_cfg["obs_space_names"]
        action_space_names = morphologycal_symmetries_cfg["action_space_names"]
        joints_order = morphologycal_symmetries_cfg["joints_order"]
        history_length = morphologycal_symmetries_cfg["history_length"]
        robot_name = morphologycal_symmetries_cfg["robot_name"]

        G, obs_reps = configure_observation_space_representations(robot_name, obs_space_names, joints_order)

        obs_space_reps = [obs_reps[n] for n in obs_space_names] * history_length
        act_space_reps = [obs_reps[n] for n in action_space_names]
        # rep_extra_obs = [
        # rep_R3, rep_R3_pseudo, trivial_rep, trivial_rep, rep_friction, rep_R3,
        # trivial_rep, trivial_rep, rep_kin_three, rep_kin_three, rep_kin_three,
        # rep_kin_three, trivial_rep, trivial_rep]

        gspace = escnn.gspaces.no_base_space(G)
        self.in_field_type = FieldType(gspace, obs_space_reps)
        self.out_field_type = FieldType(gspace, act_space_reps)

        self.critic_in_field_type = FieldType(gspace, obs_space_reps)
        self.num_replica = len(G.elements)
        self.G = G

        # one dimensional field type for critic
        critic_out_field_type = FieldType(gspace, [G.trivial_representation])

        # Construct the equivariant MLP

        self.actor = SimpleEMLP(
            self.in_field_type,
            self.out_field_type,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )

        self.critic = SimpleEMLP(
            self.critic_in_field_type,
            critic_out_field_type,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        model_parameters = filter(lambda p: p.requires_grad, self.actor.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Actor #Params: ", params)
        model_parameters = filter(lambda p: p.requires_grad, self.critic.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Critic #Params: ", params)

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update_distribution(self, observations):
        # Convert to GeometricTensor
        observations = self.in_field_type(observations)
        mean = self.actor(observations).tensor

        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act_inference(self, observations):
        observations = self.in_field_type(observations)
        actions_mean = self.actor(observations).tensor
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        critic_observations = self.critic_in_field_type(critic_observations)
        value = self.critic(critic_observations).tensor
        return value

    def export(self):
        """Export the acto-critic model as a torch.module with no Equivariant submodules."""
        torch_ac = ActorCritic(
            num_actor_obs=self.in_field_type.size,
            num_critic_obs=self.critic_in_field_type.size,
            num_actions=self.out_field_type.size,
            **self._ac_kwargs,
        )

        torch_ac.actor = self.actor.export()
        torch_ac.critic = self.critic.export()

        return torch_ac


class SimpleEMLP(EquivariantModule):
    """A simple equivariant MLP for actor-critic networks."""

    def __init__(  # noqa: D107
        self,
        in_type: FieldType,
        out_type: FieldType,
        hidden_dims=[256, 256, 256],
        bias: bool = True,
        actor: bool = True,
        activation: str = "ReLU",
    ):
        super().__init__()
        self.out_type = out_type
        gspace = in_type.gspace
        group = gspace.fibergroup

        layer_in_type = in_type
        self.net = escnn.nn.SequentialModule()
        for n in range(len(hidden_dims)):
            layer_out_type = FieldType(
                gspace,
                [group.regular_representation] * int((hidden_dims[n] / group.order())),
            )

            self.net.add_module(
                f"linear_{n}: in={layer_in_type.size}-out={layer_out_type.size}",
                escnn.nn.Linear(layer_in_type, layer_out_type, bias=bias),
            )
            self.net.add_module(f"act_{n}", self.get_activation(activation, layer_out_type))

            layer_in_type = layer_out_type

        if actor:
            self.net.add_module(
                f"linear_{len(hidden_dims)}: in={layer_in_type.size}-out={out_type.size}",
                escnn.nn.Linear(layer_in_type, out_type, bias=bias),
            )
            self.extra_layer = None
        else:
            num_inv_features = len(layer_in_type.irreps)
            self.extra_layer = torch.nn.Linear(num_inv_features, out_type.size, bias=False)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        """Forward pass through the equivariant MLP."""
        x = self.net(x)
        if self.extra_layer:
            x = self.extra_layer(x.tensor)
        return x

    @staticmethod
    def get_activation(activation: str, hidden_type: FieldType) -> EquivariantModule:
        """Returns the activation function based on the provided string."""
        if activation.lower() == "relu":
            return escnn.nn.ReLU(hidden_type)
        elif activation.lower() == "elu":
            return escnn.nn.ELU(hidden_type)
        elif activation.lower() == "lrelu":
            return escnn.nn.LeakyReLU(hidden_type)
        else:
            raise NotImplementedError

    def evaluate_output_shape(self, input_shape):
        """Returns the output shape of the model given an input shape."""
        batch_size = input_shape[0]
        return batch_size, self.out_type.size

    def export(self):
        """Exports the model to a torch.nn.Sequential instance."""
        return self.net.export()
