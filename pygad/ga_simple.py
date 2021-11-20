import torch
import pygad.torchga as torchga
import pygad
import numpy as np
import torch.nn as nn
import torch.optim as optim

from network import Network
from game import Game
from utils import *
import random
# import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network(4, 4, 10, device).to(device)
loss_fn = nn.SmoothL1Loss()
BATCH_SIZE = 10
games = []

train_rewards = []
test_rewards = []
losses = []

def fitness_func(solution, sol_idx):
    # change this to the optimize function
    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution)

    # Use the current solution as the model parameters.
    model.load_state_dict(model_weights_dict)
    g = random.sample(games)
    state = g.get_states()
    
    solution_fitness = 0
    for t in range(g.moves):
        # Select and perform an action
        action, action_dist = select_action(model, state, 0.2)
        g.attack(action)
        next_state = g.get_states()
        reward = compute_attcker_reward(state, next_state, g.get_values())
        reward = torch.tensor([reward], device=device)
        state = next_state

        # Perform one step of the optimization (on the policy network)
        # print(reward, action_dist[action])
        loss = loss_fn(reward, action_dist[action])
        test_rewards.append(test())
        losses.append(loss.item())
    
        solution_fitness += 1.0 / (loss.item() + 0.00001)

    return solution_fitness

        

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


# Create an instance of the pygad.torchga.TorchGA class to build the initial population.
torch_ga = torchga.TorchGA(model=model,
                           num_solutions=10)


# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 1000 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights
parent_selection_type = "sss" # Type of parent selection.
crossover_type = "single_point" # Type of the crossover operator.
mutation_type = "random" # Type of the mutation operator.
mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
# ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Fetch the parameters of the best solution.
best_solution_weights = torchga.model_weights_as_dict(model=model,
                                                      weights_vector=solution)
model.load_state_dict(best_solution_weights)

plot(losses, test_rewards, BATCH_SIZE)