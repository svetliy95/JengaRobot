import gym
import numpy as np
import matplotlib.pyplot as plt
import time

def get_discrete_observation(observation):
  bin = np.floor(bin_sizes / 2 * (observation / bounds + 1)).astype(int)
  bin[bin > bin_sizes - 1] = bin_sizes[bin > bin_sizes - 1] - 1
  bin[bin < 0] = 0
  Qs = Q[bin[0]][bin[1]][bin[2]][bin[3]]
  return Qs, bin

def get_action(observation, Q):
  Qs, bin = get_discrete_observation(observation)
  r = np.random.rand(1)
  if r > epsilon:
    return np.argmax(Q[bin[0]][bin[1]][bin[2]][bin[3]]), Qs, bin
  else:
    return np.argmin(Q[bin[0]][bin[1]][bin[2]][bin[3]]), Qs, bin

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# Softmax used in exploration strategy
def softmax(w, t=1.0):
    x = np.array(w)
    e = np.exp((x-np.max(x)) / t)
    dist = e / np.sum(e)
    return dist

# create and initialize the environment
env = gym.make("CartPole-v1")
observation = env.reset()

# defines
alpha = 0.5
gamma = 1
epsilon = 6
mode = 0
actions = range(env.action_space.n)
episod_length = 500
episodes = 1000

bin_sizes = None
if mode == 0:
  bin_sizes = np.array([10, 10, 10, 10])
elif mode == 1:
  bin_sizes = np.array([1, 10, 10, 10])
elif mode == 2:
  bin_sizes = np.array([1, 10, 1, 10])

bounds = [2.4, 1, 0.25, 1]

Q = np.random.rand(bin_sizes[0], bin_sizes[1], bin_sizes[2], bin_sizes[3], len(actions))


returns = np.empty(0)

# measure time
start = time.time()
for _ in range(episodes):
    episode_return = 0
    for _ in range(episod_length):
        #env.render()
        Qs, bin = get_discrete_observation(observation)

        # Softmax
        prob = softmax(Qs, epsilon)
        action = 0 if np.random.uniform(0, 1) < prob[0] else 1

        observation, reward, done, info = env.step(action)

        episode_return += reward



        Q_St_At = Q[bin[0]][bin[1]][bin[2]][bin[3]][action]
        # get expected reward of the next state
        Qs, _ = get_discrete_observation(observation)
        maxQ_St_plus_1_a = max(Qs)
        Q[bin[0]][bin[1]][bin[2]][bin[3]][action] = Q_St_At + alpha * (reward + gamma * maxQ_St_plus_1_a - Q_St_At)


        if done:
          observation = env.reset()
          break
    returns = np.append(returns, episode_return)
print("Done!")
avs = running_mean(returns, 100)

plt.figure(1)
plt.plot(returns,c='b')
plt.plot(avs, c='r')
plt.title("Return for each episode")
plt.xlabel("Episode Number")
plt.ylabel("Return (number of timesteps lasted)")
plt.legend(["Return", "Mean over last 100"], bbox_to_anchor=(0.88, 0.25), bbox_transform=plt.gcf().transFigure)
plt.savefig("./test.png")
plt.pause(1e-20)



env.close()


end = time.time()

print(end-start)


