import gymnasium as gym
import numpy as np
import random

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
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

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, num_generations, n_episodes):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.n_episodes = n_episodes

    def initialize_population(self):
        return [np.random.rand(9) * 4.0 for _ in range(self.population_size)]

    def fitness(self, individual):
        vertical_params = individual[0:3]
        horizontal_params = individual[3:6]
        angle_params = individual[6:9]
        
        env = gym.make("LunarLander-v3", render_mode=None, gravity=-10.0)
        vertical_pid = PIDController(*vertical_params)
        horizontal_pid = PIDController(*horizontal_params)
        angle_pid = PIDController(*angle_params)
        
        total_rewards = []
        
        for episode in range(self.n_episodes):
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

        env.close()
        return np.mean(total_rewards)

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            mutation_value = np.random.randn() * 0.1
            individual[np.random.randint(0, 9)] += mutation_value

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(0, 9)
            return np.concatenate([parent1[:point], parent2[point:]])
        return parent1 if np.random.rand() < 0.5 else parent2

    def run(self):
        population = self.initialize_population()

        for generation in range(self.num_generations):
            population_fitness = [self.fitness(ind) for ind in population]
            sorted_indices = np.argsort(population_fitness)
            population = [population[i] for i in sorted_indices]
            best_individual = population[-1]
            print(f"Generation {generation}, Best Fitness: {population_fitness[sorted_indices[-1]]}")

            next_generation = population[-10:]

            while len(next_generation) < self.population_size:
                parent1, parent2 = random.sample(population[-20:], 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                next_generation.append(child)

            population = next_generation

        return best_individual

def play_with_best_params(params, n_episodes=1, render_mode='human'):
    env = gym.make("LunarLander-v3", render_mode=render_mode, gravity=-10.0)
    
    vertical_pid = PIDController(*params[:3])
    horizontal_pid = PIDController(*params[3:6])
    angle_pid = PIDController(*params[6:9])
    
    total_rewards = []

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
        print(f"Episode {episode}: {total_reward}")
        print(f"Landing on ({coord_x:.2f},{coord_y:.2f}) with velocity = {velocity_y:.2f}")

    env.close()
    print(f"Mean Reward over {n_episodes} episode(s): {np.mean(total_rewards)}")

if __name__ == "__main__":
    # Genetic Algorithm Configuration
    ga = GeneticAlgorithm(
        population_size=100,
        mutation_rate=0.1,
        crossover_rate=0.7,
        num_generations=50,
        n_episodes=5
    )

    # Running the Genetic Algorithm to get the best PID parameters
    best_pid_params = ga.run()
    print("Best PID Parameters (vertical, horizontal, angle):")
    print("Vertical PID:", best_pid_params[:3])
    print("Horizontal PID:", best_pid_params[3:6])
    print("Angle PID:", best_pid_params[6:9])

    # Visualize the performance of the best parameters
    print("\nPlaying with the best parameters...\n")
    play_with_best_params(best_pid_params, n_episodes=5)