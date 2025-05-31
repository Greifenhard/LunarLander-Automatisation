import gymnasium as gym
import numpy as np

# Parameter der genetischen Algorithmen
population_size = 100
num_generations = 50
mutation_rate = 0.1

# Trainingsumgebung initialisieren (ohne Render-Modus)
train_env = gym.make('LunarLander-v3', render_mode = None)

def evaluate(individual, env):
    """Bewertet einen Einzelnen auf einer Episode."""
    total_reward = 0
    state, _ = env.reset()
    for _ in range(1000):  # Maximale Schritte pro Episode, da es zu unendlichen Schritten kommen kann...
        action = np.argmax(np.dot(state, individual))  # Beobachtungsvektor kommt zuerst
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

def mutate(individual):
    """Führt Mutationen am Individuum durch."""
    for i in range(individual.shape[0]):
        for j in range(individual.shape[1]):
            if np.random.rand() < mutation_rate:
                individual[i, j] += np.random.normal(0, 1)
    return individual

def crossover(parent1, parent2):
    """Kombiniert zwei Eltern zu einem Kind."""
    crossover_point = np.random.randint(parent1.shape[0])
    child = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    return child

def genetic_algorithm(env):
    # Initialisierung der Bevölkerung mit zufälligen Individuen
    population = [np.random.rand(env.observation_space.shape[0], env.action_space.n) for _ in range(population_size)]

    for generation in range(num_generations):
        # Bewertung der aktuellen Population
        scores = [evaluate(individual, env) for individual in population]
        
        # Auswahl der besten Individuen
        sorted_indices = np.argsort(scores)[::-1]  # Sortierung nach Bewertung
        population = [population[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Ausgeben des besten Ergebnisses der Generation
        print(f"Generation {generation}: Best Score = {scores[0]}")
        
        # Selektion und Erzeugung der nächsten Generation
        new_population = population[:10]  # Beste 10% der Population übernehmen
        while len(new_population) < population_size:
            parent_indices = np.random.choice(20, 2, replace=False)  # Auswahl der Elternindizes
            parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
            child = crossover(parent1, parent2)  # Kreuzung
            child = mutate(child)  # Mutation
            new_population.append(child)

        population = new_population

    # Ausgabe des besten Individuums am Ende
    best_individual = population[0]

    # Speichern der besten Strategie
    np.save('best_strategy.npy', best_individual)

def load_and_run(n_times=10, render_mode="human"):
    # Beste Strategie laden
    best_strategy = np.load('best_strategy.npy')
    play_env = gym.make('LunarLander-v3', render_mode=render_mode, gravity=-11.0)
    
    
    for _ in range(n_times):
        # Benutze die geladene Strategie
        state, _ = play_env.reset()

        total_reward = 0
        for _ in range(1000):
            action = np.argmax(np.dot(state, best_strategy))
            state, reward, terminated, truncated, _ = play_env.step(action)            
            total_reward += reward
            
            if terminated or truncated:
                break
            
        print(total_reward)
    play_env.close()

#genetic_algorithm(train_env)  # Berechne und speichere den besten genetischen Algorithmus
load_and_run(300)  # Lade den besten Algorithmus und spiele damit