import pybullet_envs
import gym
import numpy
from sac import Agent
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Average of previos 100 scores')
    plt.savefig(figure_file)


if __name__ == '__main__':
    env_id = 'InvertedPendulumBulletEnv-v0'
    env = gym.make(env_id)
    agent = Agent(alpha=0.003, beta=0.003, reward_scale=2, env_id=env_id,
                  input_dims=env.observation_space.shape, tau=0.005,
                  env=env, batch_size=256, layer1_size=256, layer2_size=256,
                  n_actions=env.action_space.shape[0])
    n_games = 250
    filename = env_id+ '_' +str(n_games)+'games_scale'+str(agent.scale)+'.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_madels()
        env.render(mode='human')

    steps=0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()

        while not done:
            action = agent.choose_actions(observation)
            observation_, reward, done, info = env.step(action)
            steps += 1
            score += reward
            agent.remember(observation,action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode', i, 'score %.1f' % avg_score,
              'training 100 games avg %.1f' % avg_score,
              'steps %d' % steps, env_id, 'scale', agent.scale)
    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)











