import gymnasium as gym
import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp    # Volle Kraft vorraus
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def reset(self):
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

def evaluate_pid(n_episodes=5, verbose=0):
    env = gym.make("LunarLander-v3", render_mode=None, gravity=-10.0)
    vertical_pid   = PIDController(Kp=4.0, Ki=2.0, Kd=0.5)
    horizontal_pid = PIDController(Kp=0.5, Ki=0.7, Kd=0.7)
    angle_pid      = PIDController(Kp=2.0, Ki=0.7, Kd=0.3)
    total_rewards  = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        vertical_pid.reset()
        horizontal_pid.reset()
        angle_pid.reset()
        total_reward = 0
        done = False
        
        while not done:
            
            coord_x, coord_y, velocity_x, velocity_y, angle, velocity_angle, left_foot, right_food = state
            
            target_y = 0.0
            target_x = 0.0
            target_angle = 0.0

            y_error = target_y - coord_y - velocity_y * 2
            x_error = target_x + coord_x + velocity_x * 2
            angle_error = target_angle - angle - velocity_angle
            
            dt = 1.0 / 50.0 

            y_action = vertical_pid.update(y_error, dt)
            x_action = horizontal_pid.update(x_error, dt)
            theta_action = angle_pid.update(angle_error, dt)
            
            if verbose:
                print("y: ", coord_y, "v: ", velocity_y, end=" ")
                print("Fehler: ", y_error, "Output: ", y_action)
                print("x: ",coord_x, "v: ", velocity_x, end=" ")
                print("Fehler: ", x_error, "Output: ", x_action)
                print()
            
            vertical_action = 2 if y_action > 0 else 0
            horizontal_action = 1 if x_action > 0 else 3
            angle_action = 1 if theta_action > 0 else 3

            action = vertical_action
            if vertical_action == 0:
                action = horizontal_action if abs(x_action) > abs(theta_action) else angle_action

            state, reward, done, _, _ = env.step(action)
            if left_foot and right_food and abs(velocity_y) < 0.005:
                done = True
            total_reward += reward
            
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: {total_reward}")
        print(f"Landing on ({coord_x:.2f},{coord_y:.2f}) with velocity = {velocity_y:.2f}")
        print()
    env.close()
    print(f"Mean Score through {episode + 1} episodes: {np.mean(total_rewards)}")

evaluate_pid(n_episodes=15)