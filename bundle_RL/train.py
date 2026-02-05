import torch
import torch.optim as optim
from bundle_env import BundleDualEnv
from policy import LambdaPolicy

env = BundleDualEnv(problem_data={}, K=10, T=20)

policy = LambdaPolicy(env.state_dim, env.action_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-4)

gamma = 0.99

for episode in range(500):
    state = torch.tensor(env.reset()).float()
    log_probs = []
    rewards = []

    done = False
    while not done:
        lambdas = policy(state)
        dist = torch.distributions.Categorical(lambdas)
        action = dist.sample()

        next_state, reward, done, info = env.step(lambdas.detach().numpy())

        log_probs.append(dist.log_prob(action))
        rewards.append(reward)

        state = torch.tensor(next_state).float()

    # returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = 0
    for log_p, R in zip(log_probs, returns):
        loss -= log_p * R

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 10 == 0:
        print(f"Episode {episode}, final dual value {info['phi']:.4f}")
