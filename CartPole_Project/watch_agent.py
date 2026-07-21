import gymnasium as gym
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('models/cartpole_dqn_model.h5')

# Initialize the environment with human rendering
env = gym.make('CartPole-v1', render_mode='human')

# Run the test episode
state, _ = env.reset()
done = False
total_reward = 0

print("Starting simulation...")

while not done:
    q_values = model(np.array([state])).numpy()
    action = np.argmax(q_values[0])
    
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    total_reward += reward

print(f"Simulation finished! Total Score: {total_reward}")
env.close()