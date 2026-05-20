import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Beta
from torch.distributions.utils import clamp_probs

from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SimpleBundleExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        cuts_shape = observation_space["cuts"].shape
        pi_shape = observation_space["pi"].shape
        self.cuts_dim = cuts_shape[0] * cuts_shape[1]
        self.pi_dim = pi_shape[0]

        self.net = nn.Sequential(
            nn.Linear(self.cuts_dim + self.pi_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        cuts = observations["cuts"].flatten(1)
        pi = observations["pi"]
        x = torch.cat([cuts, pi], dim=1)
        return self.net(x)


class BundleActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 eta_scale=1.0, hidden_dim=128, *args, **kwargs):
        self.eta_scale = eta_scale
        self.hidden_dim = hidden_dim
        self.K = action_space.shape[0] - 1

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=SimpleBundleExtractor,
            features_extractor_kwargs=dict(features_dim=hidden_dim),
            net_arch=dict(pi=[hidden_dim, hidden_dim], vf=[hidden_dim, hidden_dim]),
            *args, **kwargs
        )

        self._lambda_alpha = None
        self._eta_alpha = None
        self._eta_beta = None
        self._valid_mask = None

    def _build_actor(self, features_dim):
        self.shared_net = nn.Sequential(
            nn.Linear(features_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.lambda_net = nn.Linear(self.hidden_dim, self.K)
        self.eta_net = nn.Linear(self.hidden_dim, 2)

    def _get_distribution(self, obs):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        shared_repr = self.shared_net(latent_pi)

        raw_alpha = self.lambda_net(shared_repr)
        alpha_pre_mask = F.softplus(raw_alpha) + 1e-6

        self._valid_mask = obs["valid_mask"]
        self._lambda_alpha = alpha_pre_mask * self._valid_mask + 1e-4

        eta_params = self.eta_net(shared_repr)
        self._eta_alpha = F.softplus(eta_params[:, 0]) + 1e-6
        self._eta_beta = F.softplus(eta_params[:, 1]) + 1e-6

        dirichlet_dist = Dirichlet(self._lambda_alpha, validate_args=False)
        beta_dist = Beta(self._eta_alpha, self._eta_beta, validate_args=False)

        return dirichlet_dist, beta_dist

    def forward(self, obs, deterministic=False):
        dirichlet_dist, beta_dist = self._get_distribution(obs)

        if deterministic:
            lambda_action = dirichlet_dist.mode()
        else:
            lambda_action = dirichlet_dist.rsample()

        eta_action = beta_dist.rsample() * self.eta_scale

        action = torch.cat([lambda_action, eta_action.unsqueeze(-1)], dim=-1)

        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        value = self.value_net(latent_vf)

        return action, value

    def evaluate_actions(self, obs, actions):
        dirichlet_dist, beta_dist = self._get_distribution(obs)

        lambdas = actions[:, :-1]
        eta = actions[:, -1] / self.eta_scale

        log_prob_lambda = dirichlet_dist.log_prob(lambdas)
        log_prob_eta = beta_dist.log_prob(clamp_probs(eta))
        log_prob = log_prob_lambda + log_prob_eta

        entropy_lambda = dirichlet_dist.entropy()
        entropy_eta = beta_dist.entropy()
        entropy = entropy_lambda + entropy_eta

        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        value = self.value_net(latent_vf)

        return value, log_prob, entropy
