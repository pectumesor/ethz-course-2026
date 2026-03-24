"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""

# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):
    """Predicts action chunks with an MSE loss.

    A simple MLP that maps a state vector to a flat action chunk
    (chunk_size * action_dim) and reshapes to (B, chunk_size, action_dim).
    """

    def __init__(
        self, state_dim: int, action_dim: int, chunk_size: int, d_model:int, depth:int,
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, chunk_size=chunk_size)

        layer_sizes = [d_model] * (depth - 1) + [chunk_size * action_dim]
        curr_layer = state_dim

        layers = nn.ModuleList()

        for size in layer_sizes:
            if size == chunk_size * action_dim:
                layers.append(nn.Linear(curr_layer, size))
            else:
                layers.append(nn.Linear(curr_layer,size))
                layers.append(nn.ReLU())
                curr_layer = size

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""
        x = self.mlp(state)
        return x.reshape(state.shape[0], self.chunk_size, self.action_dim)

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        pred_actions = self.sample_actions(state)
        return torch.nn.functional.mse_loss(pred_actions, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(state)

# TODO: Students implement MultiTaskPolicy here.
class MultiTaskPolicy(BasePolicy):
    """Goal-conditioned policy for the multicube scene."""

    def __init__(
        self, state_dim: int, action_dim: int, chunk_size: int, d_model:int,
        depth:int, hidden_size: int = 128, latent_dim: int = 32
    ) -> None:
        super().__init__(state_dim=state_dim, action_dim=action_dim, chunk_size=chunk_size)

        self.latent_dim = latent_dim

        # Encode cubes position: Learns high level features such as distance from ee, what and where cubes are, etc.
        self.cube_encoder = nn.Sequential( nn.Linear(3, hidden_size // 2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size // 2, hidden_size) 
                                        ) 
        
        # Goal Conditioning features: State Goal and Goal Position (dim: 3+3=6)
        self.goal_cond_encoder = nn.Sequential(nn.Linear(6, hidden_size // 2),
                                                nn.ReLU(),
                                                nn.Linear(hidden_size // 2, hidden_size),
                                                nn.ReLU())

        # Encodes robot states 3 (ee pos) + 1 (gripper)  = 4
        self.robot_encoder = nn.Sequential(nn.Linear(4, hidden_size // 2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size // 2, hidden_size) 
                                        ) 
        
        condition_dim = 6 * hidden_size # Condition variable = [goal_feat, robot_feat, target_feat, red_feat, blue_feat, green_feat]

        self.condition_head = nn.Sequential( nn.Linear(condition_dim, d_model),
                                             nn.LayerNorm(d_model),
                                             nn.ReLU(),
                                             nn.Dropout(0.1)
                                            )
        
        # Variational Encoder
        curr_layer = d_model
        layer_sizes = [d_model] * depth
        encoder_layers = nn.ModuleList()
        build_layer_block(encoder_layers, layer_sizes, curr_layer, "encoder")
        self.encoder = nn.Sequential(*encoder_layers)

        self.mu = nn.Linear(d_model, latent_dim)
        self.log_var = nn.Linear(d_model, latent_dim)

        # Variational Decoder
        layer_sizes = [d_model] * (depth - 1)
        curr_layer = d_model + latent_dim
        decoder_layers = nn.ModuleList()
        build_layer_block(decoder_layers, layer_sizes, curr_layer, "decoder")
        decoder_layers.append(nn.Linear(d_model, chunk_size * action_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def prior(self, cond_feat: torch.Tensor) -> torch.Tensor:
        h = self.prior_net(cond_feat)
        mu = self.prior_mu(h)
        log_var = self.prior_logvar(h)
        return mu, log_var


    def split_state(self, state: torch.Tensor) ->tuple[torch.Tensor,...]:

        ee_pos = state[:, 0:3]
        gripper = state[:, 3:4]
        state_goal = state[:,4:7]
        goal_pos = state[:, 7:10]
        red_cube = state[:, 10:13]
        green_cube = state[:, 13:16]
        blue_cube = state[:, 16:19]

        return ee_pos, gripper, state_goal, goal_pos, red_cube, green_cube, blue_cube
    

    def encode_condition_variable(self, state: torch.Tensor) -> torch.Tensor:

        ee_pos, gripper, state_goal, goal_pos, red_cube, green_cube, blue_cube = self.split_state(state)

        red_rel = red_cube - ee_pos
        green_rel = green_cube - ee_pos
        blue_rel = blue_cube - ee_pos

        red_feat = self.cube_encoder(red_rel)
        green_feat = self.cube_encoder(green_rel)
        blue_feat = self.cube_encoder(blue_rel)

        goal_feat = self.goal_cond_encoder(torch.cat([state_goal, goal_pos - ee_pos], dim=-1))
        robot_feat = self.robot_encoder(torch.cat([ee_pos, gripper], dim=-1))

        # Hard Mask on cubes to encode real target
        cubes = torch.stack([red_feat, green_feat, blue_feat], dim=1)
        target_feat = (cubes * state_goal.unsqueeze(-1)).sum(dim=1)

        cond_feat = torch.cat([robot_feat, goal_feat, target_feat, red_feat, green_feat, blue_feat], dim=-1)

        return self.condition_head(cond_feat)
    
    def encode(self, cond_feat: torch.Tensor) -> tuple[torch.Tensor, ...]:

        h = self.encoder(cond_feat)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var
    
    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var) 
        epsilon = torch.randn_like(std)
        return mu + epsilon * std
    
    def decode(self, z: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cond_feat, z], dim=-1)
        return self.decoder(x)

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted action chunk of shape (B, chunk_size, action_dim)."""

        cond_feat = self.encode_condition_variable(state)
        mu, log_var = self.encode(cond_feat)
        z = self.sample_z(mu, log_var)
        x = self.decode(z, cond_feat)   
        return x.view(cond_feat.shape[0], self.chunk_size, self.action_dim), mu, log_var

    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:  
        pred_actions, mu, log_var = self.forward(state)

        recon_loss = nn.functional.mse_loss(pred_actions, action_chunk, reduction='mean')

        kl_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

        beta = 1e-4
        return recon_loss + beta * kl_loss

    def sample_actions(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        action, _, _ = self.forward(state)
        return action.view(state.shape[0], self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]

def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    d_model: int,
    depth: int,
) -> BasePolicy:
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            # TODO: Build with your chosen specifications
            chunk_size=chunk_size,
            d_model=d_model,
            depth=depth
        )
    raise ValueError(f"Unknown policy type: {policy_type}")


## --- Utility Function --- ##

def build_layer_block(layers, layer_sizes: list[int], curr_layer: int, layer_type: str):

    if layer_type == "encoder":
        for size in layer_sizes:
            layers.append(nn.Linear(curr_layer,size))
            layers.append(nn.ReLU())
            curr_layer = size
    else:
        for size in layer_sizes:
            layers.append(nn.Linear(curr_layer,size))
            layers.append(nn.LayerNorm(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            curr_layer = size
