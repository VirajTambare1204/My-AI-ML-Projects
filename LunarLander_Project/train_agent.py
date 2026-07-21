import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random

# 1. Initialize Environment
env = gym.make('LunarLander-v3')
num_states = env.observation_space.shape[0] 
num_actions = env.action_space.n 

# 2. Hyperparameters
learning_rate = 0.001
gamma = 0.99           
epsilon = 1.0          
epsilon_min = 0.01     
epsilon_decay = 0.995  
batch_size = 64 # Increased batch size for the harder environment

# 3. Replay Buffer
memory = deque(maxlen=100000) # Increased memory pool

# 4. Build a slightly larger Q-Network
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(num_states,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_actions, activation='linear') 
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

# 5. Define the Training Step
def train_step():
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    states = np.array([transition[0] for transition in batch])
    actions = np.array([transition[1] for transition in batch])
    rewards = np.array([transition[2] for transition in batch])
    next_states = np.array([transition[3] for transition in batch])
    dones = np.array([transition[4] for transition in batch])

    q_values = model(states).numpy() 
    next_q_values = model(next_states).numpy()

    for i in range(batch_size):
        if dones[i]:
            q_values[i][actions[i]] = rewards[i]
        else:
            q_values[i][actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])
            
    model.fit(states, q_values, epochs=1, verbose=0)

# 6. Main Training Loop
episodes = 2000

for e in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            q_values = model(np.array([state])).numpy()
            action = np.argmax(q_values[0])
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        memory.append((state, action, reward, next_state, done))
        
        state = next_state
        total_reward += reward
        
        train_step()
        
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        
    print(f"Episode: {e+1}/{episodes} | Score: {total_reward:.2f} | Epsilon: {epsilon:.2f}")
    
    # Lunar Lander is officially solved at a score of 200
    if total_reward >= 200:
        print("Solved!")
        model.save('models/lunar_lander_dqn.h5')
        break

env.close()