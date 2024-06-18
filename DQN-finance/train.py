# Improvement: Steps

from dqn.dqn import *
from models.StockEnv import StockEnv
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import os
from tqdm import tqdm
import torch


def resize_images(directory="train/", output_size=(417, 300)):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            with Image.open(img_path) as img:
                img = img.resize(output_size,
                                 Image.Resampling.LANCZOS)
                img.save(img_path)


def train_dqn(gamma=0.99, lr=1e-3, min_episodes=1, eps=1, eps_decay=0.99, eps_min=0.02, update_step=4, batch_size=4096,
              update_repeats=3, num_episodes=3000, seed=42, max_memory_size=110000, lr_gamma=0.9, lr_step=100,
              measure_step=30, save_episodes=100):
    env = StockEnv()
    test_env = StockEnv(test=True)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Q_1 = QNetworkCNN(action_dim=env.action_space.n).to(device)
    Q_2 = QNetworkCNN(action_dim=env.action_space.n).to(device)
    update_parameters(Q_1, Q_2)

    for param in Q_2.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    memory = Memory(max_memory_size)
    performance = []
    best_reward = -np.inf
    best_weights = None

    for episode in tqdm(range(num_episodes)):
        states = env.reset()
        total_reward = 0

        # No more While not done: loop because we take all states at once
        actions = select_actions(Q_2, env, states, eps)
        rewards = env.calculate_rewards(actions)

        memory.update(states, actions, rewards)
        total_reward += sum(rewards)

        # Clear memory
        del states, actions, rewards
        if episode % measure_step == 0 and episode >= min_episodes:
            current_performance, _ = evaluate(Q_2, test_env, 50)

            performance.append([episode, current_performance])
            if current_performance > best_reward:
                best_reward = current_performance
                best_weights = Q_1.state_dict()
                torch.save(Q_1.state_dict(), f'weights/best_weights_last_epoch_{episode}.pth')

            print("Average Reward: ", current_performance)
        if episode % save_episodes == 0:
            torch.save(Q_1.state_dict(), f'weights/weights_last_epoch_{episode}.pth')

        if episode >= min_episodes and episode % update_step == 0:
            for _ in range(update_repeats):
                train(batch_size, Q_1, Q_2, optimizer, memory, gamma)
            update_parameters(Q_1, Q_2)

        scheduler.step()
        eps = max(eps * eps_decay, eps_min)

    if best_weights is not None:
        torch.save(best_weights, 'weights_best.pth')

    return performance





if __name__ == '__main__':
    # # Start profiling
    # profiler = cProfile.Profile()
    # profiler.enable()

    Q_1, performance = train_dqn()
    # torch.save(Q_1.state_dict(), 'weights.pth')
    # resize_images("test/")


    # # Stop profiling
    # profiler.disable()
    # # Print profiling results
    # profiler.print_stats()


