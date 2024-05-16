import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from cec2017.functions import f3, f19

DIM = 10  # wymiarowość problemu
max_evaluations = 1000
max_num_iterations = 1000
learning_rate = 1e-10

def one_plus_one_ES(objective_function, sigma, max_evaluations):
    # Inicjalizacja
    best_solution = np.random.uniform(100, 100, size=DIM)  # Losowy początkowy wektor
    best_fitness = objective_function(best_solution)
    evaluations = 1
    fitness_history = [best_fitness]

    while evaluations < max_evaluations:
        # Mutacja
        mutation = np.random.normal(0, sigma, size=DIM)
        new_solution = best_solution + mutation

        # Ograniczenie wartości do zakresu [-100, 100]
        new_solution = np.clip(new_solution, -100, 100)

        # Ocena nowego rozwiązania
        new_fitness = objective_function(new_solution)
        evaluations += 1

        # Aktualizacja najlepszego rozwiązania
        if new_fitness < best_fitness:
            best_solution = new_solution
            best_fitness = new_fitness

        # Zapisanie historii wartości funkcji
        fitness_history.append(best_fitness)

    return best_solution, best_fitness, fitness_history

def plot_convergence(objective_function, name):
    sigmas = [0.5, 1.0, 2.0]
    evaluations = range(1, max_evaluations + 1)

    plt.figure(figsize=(10, 8))

    for sigma in sigmas:
        best_fitness_histories = []

        for _ in range(10):  # Przeprowadź 10 iteracji
            _, _, fitness_history = one_plus_one_ES(objective_function, sigma, max_evaluations)
            best_fitness_histories.append(fitness_history)

        averaged_fitness_history = np.mean(best_fitness_histories, axis=0)

        plt.plot(evaluations, averaged_fitness_history, label=f'sigma={sigma}')

    plt.xlabel('Evaluations')
    plt.ylabel('Fitness')
    plt.title(f'Convergence Plot for {name}')
    plt.xscale('log')  # Ustawienie skali logarytmicznej dla osi x
    plt.xticks([10**i for i in range(5)], [f'$10^{i}$' for i in range(5)])  # Ustawienie stałych odstępów na osi x
    plt.grid(True)
    plt.legend()
    plt.show()

def gradient_descent(objective_function, x0, max_num_iterations):
    x_current = np.array(x0, dtype=float)
    gradient_q = grad(objective_function)

    q_values = []

    for i in range(max_num_iterations):
        q_value = objective_function(x_current)
        q_values.append(q_value)

        grad_value = gradient_q(x_current)
        x_next = x_current - learning_rate * grad_value
        x_current = x_next

    return q_values

def gradient_descent_19(objective_function, x0, max_num_iterations):
    x_current = np.array(x0, dtype=float)
    q_values = []

    for i in range(max_num_iterations):
        q_value = objective_function(x_current)
        q_values.append(q_value)

        # Oblicz ręcznie gradient dla funkcji objective_function (f19)
        grad_value = compute_gradient(objective_function, x_current)
        x_next = x_current - learning_rate * grad_value
        x_current = x_next

    return q_values

def compute_gradient(objective_function, x):
    h = 1e-6  # Mała wartość różniczki
    gradient = np.zeros_like(x)

    for i in range(len(x)):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        grad_i = (objective_function(x_plus_h) - objective_function(x)) / h
        gradient[i] = grad_i

    return gradient

if __name__ == "__main__":
    # Badanie zbieżności dla funkcji f3
    print("Results for f3:")
    plot_convergence(f3, 'f3')

    # Badanie zbieżności dla funkcji f19
    print("\nResults for f19:")
    plot_convergence(f19, 'f19')

    # Początkowe wartości x
    x_start_values = np.random.uniform(100, 100, size=DIM)

    # Wykonaj algorytm gradientu prostego dla funkcji f3
    q_values_f3 = gradient_descent(f3, x_start_values, max_num_iterations)

    # Wykres zbieżności dla f3
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, max_num_iterations + 1), q_values_f3,
                 label='Objective Function Value (f3)')
    plt.xlabel('Iteration')
    plt.ylabel('q(x)')
    plt.title('Convergence of Gradient Descent (f3)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Wykonaj algorytm gradientu prostego dla funkcji f19
    q_values_f19 = gradient_descent_19(f19, x_start_values, max_num_iterations)

    # Wykres zbieżności dla f19
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, max_num_iterations + 1), q_values_f19,
                 label='Objective Function Value (f19)')
    plt.xlabel('Iteration')
    plt.ylabel('q(x)')
    plt.title('Convergence of Gradient Descent (f19)')
    plt.grid(True)
    plt.legend()
    plt.show()
