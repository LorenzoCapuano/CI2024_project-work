Project-work
For the project, expressions were represented using a recursive structure of nodes, which can be of two types: "operation" and "termination".
	Operations are characterized by the number of input arguments and the operation that processes them to produce a result. The inputs of an operation are other nodes.
	Terminations can either be constants or variables.
Operations are performed on vectors rather than individual elements. Therefore, given an instance of the problem and an expression, the expression is evaluated only once. This allows storing the result of each operation that makes up the expression, so it can be used later. In fact, due to the nature of the operations that modify the expressions, only part of the expression is changed, and it is not necessary to recalculate the result of the operations that are unaffected by the modification.

Initialization
Half of the population is initialized using the Full method, and the other half using the Grow method. The elements that make up the initialized population are the building blocks used to construct the expressions of future generations. Therefore, it is essential to initialize a population with a diverse set of operations and constants to effectively model the problem data. The available elements during initialization are defined in a pool of operations and constants.

Selection
Parent selection is tournament selection (tournament size 2) with a fitness hole which favors shorter expressions.
Survivor selection, the fittest individual of the offspring pass to the next generation (Generational model)

Crossover
Given two expressions, a sub-expression is identified in each. These two sub-expressions are then swapped, resulting in two new expressions. 
Since the two new expressions originate from previously evaluated expressions, we only need to recompute a subset of their operations. (red ones)

Mutations
The mutation techniques used are hoisting and collapse. Given an expression, one of its sub-expressions is hoisted, while the rest of the expression is collapsed. The mutations occur simultaneously, generating two new expressions. As with crossover, these modifications do not require a complete reevaluation of the newly obtained expressions. I tried implementing other mutation techniques, such as point mutation, but they led to worst performance.

Diversity
To promote diversity, I applied a strategy that combines elements of segregation and extinction. N independent populations evolve for a number of iterations (epoch). Then, an extinction step is applied to each, reducing their size. The remaining populations are merged, and from this combined population, n copies are created. These copies then evolve separately, restarting the cycle.
Before resuming evolution, any empty slots in each population are filled with randomly generated expressions.

Allopatric selection
Since genetic operators produce multiple expressions, allopatric selection can be applied. To determine whether this type of selection provides benefits in this case, I solved the same problem multiple times using the two different approaches and plotted the distribution of the MSA of the best final individuals.
The distributions turned out to be very similar. Due to the small statistical sample (200 runs), caused by the limited computational power available to me, it is not possible to draw a definitive conclusion. In the final version, I chose to use allopatric selection.

Parameters
Strategy = (μ,λ)
Population size = 200 (µ)
Number of generations per epoch = 15
Number of epoch = 7
Probability of crossover = 0.9
Probability of mutation = 0.1
λ = 4
Offspring size = 800
Maximum initial depth = 3
Fitness hole = 0.1 (The probability in a tournament of selecting the smaller individual rather than the best one)

For the fifth problem, I multiplied the y values by 1e10, then performed symbolic regression, and finally applied a division by 1e10 to the obtained formula.
