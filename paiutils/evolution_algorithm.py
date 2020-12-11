"""
Author: Travis Hammond
Version: 12_10_2020
"""


from multiprocessing.pool import ThreadPool
import numpy as np


class Fitness:
    """Fitness contains functions that return functions that
       calculate fitness.
    """

    @staticmethod
    def match_mse(target_genome, variable_size=False):
        """Creates a fitness_func that computes
           the the mean squared error of the target
           genome and the offsprings' genome.
        params:
            target_genome: A list or numpy array that is the
                         target genome
            variable_size: A boolean, which determines if the
                           genome size can change
        return: A fitness function
        """
        target_genome = np.array(target_genome)
        if variable_size:
            def fitness_func(offspring):
                errors = []
                for genome in offspring:
                    error = (genome.shape[0]-target_genome.shape[0])**2
                    error += (genome[:target_genome.shape[0]] -
                              target_genome[:genome.shape[0]])**2
                    errors.append(error.mean())
                return np.array(errors)
        else:
            def fitness_func(offspring):
                error = (offspring - target_genome)**2
                error = error.reshape((error.shape[0], -1)).mean(axis=1)
                return error
        return fitness_func


class Selection:
    """Selection contains functions that return functions
       that select offspring based on fitness values.
    """

    @staticmethod
    def select_highest(variable_size=False):
        """Creates a selection function that selects offspring
           with the highest fitness value.
        params:
            variable_size: A boolean, which determines if the
                           genome size can change
        return: A selection function
        """
        if variable_size:
            def selection_func(offspring, fitness_values, selection_size):
                new_offspring = []
                for ndx in fitness_values.argsort()[-selection_size:]:
                    new_offspring.append(offspring[ndx])
                return new_offspring
        else:
            def selection_func(offspring, fitness_values, selection_size):
                return offspring[fitness_values.argsort()[-selection_size:]]
        return selection_func

    @staticmethod
    def select_lowest(variable_size=False):
        """Creates a selection function that selects offspring
           with the lowest fitness value.
        params:
            variable_size: A boolean, which determines if the
                           genome size can change
        return: A selection function
        """
        if variable_size:
            def selection_func(offspring, fitness_values, selection_size):
                new_offspring = []
                for ndx in fitness_values.argsort()[:selection_size]:
                    new_offspring.append(offspring[ndx])
                return new_offspring
        else:
            def selection_func(offspring, fitness_values, selection_size):
                return offspring[fitness_values.argsort()[:selection_size]]
        return selection_func


class Crossover:
    """Crossover contains functions that return functions
       that mix parent genes and return offspring.
    """

    @staticmethod
    def dual(formula=lambda a, b: (a + b)/2):
        """Creates a crossover function that mixes two parents
           to create one offspring.
        params:
            formula: A function that takes Parent A and Parent B
                     and returns one Child
        return: A crossover function
        """
        def crossover_func(parents, num_offspring):
            replace = parents.shape[0] < num_offspring
            parents_a = np.random.choice(np.arange(parents.shape[0]),
                                         size=num_offspring, replace=replace)
            parents_b = np.random.choice(np.arange(parents.shape[0]),
                                         size=num_offspring, replace=replace)
            return formula(parents[parents_a], parents[parents_b])

        return crossover_func

    @staticmethod
    def triple(formula=lambda a, b, c: (a + b + c)/3):
        """Creates a crossover function that mixes three parents
           to create one offspring.
        params:
            formula: A function that takes Parent A, Parent B, and Parent C
                     and returns one Child
        return: A crossover function
        """
        def crossover_func(parents, num_offspring):
            replace = parents.shape[0] < num_offspring
            parents_a = np.random.choice(np.arange(parents.shape[0]),
                                         size=num_offspring, replace=replace)
            parents_b = np.random.choice(np.arange(parents.shape[0]),
                                         size=num_offspring, replace=replace)
            parents_c = np.random.choice(np.arange(parents.shape[0]),
                                         size=num_offspring, replace=replace)
            return formula(parents[parents_a], parents[parents_b],
                           parents[parents_c])

        return crossover_func

    @staticmethod
    def population_avg():
        """Creates a crossover function that averages all the parents to
           create all the exact same offspring.
        return: A crossover function
        """
        def crossover_func(parents, num_offspring):
            return np.repeat([np.mean(parents, axis=0)], num_offspring, axis=0)

        return crossover_func

    @staticmethod
    def population_shuffle():
        """Creates a crossover function that randomly
           shuffles all the genes between all the parents.
        return: A crossover function
        """
        def crossover_func(parents, num_offspring):
            assert parents.ndim == 2, (
                'Population shuffle only works with inputs '
                'that have 2 dimensions.'
            )
            num_genes = parents.shape[1]
            old_parents = parents
            parents = parents.T.copy()
            for gene in range(num_genes):
                np.random.shuffle(parents[gene])
            parents = np.vstack([old_parents, parents.T])
            replace = parents.shape[0] < num_offspring
            indexes = np.random.choice(np.arange(parents.shape[0]),
                                       size=num_offspring, replace=replace)
            return parents[indexes]

        return crossover_func

    @staticmethod
    def single(variable_size=False):
        """Creates a crossover function that does not perform
           any crossover, but instead creates a child from a
           single parent. (Parents may produce more than one child)
        params:
            variable_size: A boolean, which determines if the
                           genome size can change
        return: A crossover function
        """
        if variable_size:
            def crossover_func(parents, num_offspring):
                replace = len(parents) < num_offspring
                indexes = np.random.choice(np.arange(len(parents)),
                                           size=num_offspring, replace=replace)
                new_offspring = []
                for ndx in indexes:
                    new_offspring.append(parents[ndx])
                return new_offspring
        else:
            def crossover_func(parents, num_offspring):
                replace = parents.shape[0] < num_offspring
                indexes = np.random.choice(np.arange(parents.shape[0]),
                                           size=num_offspring, replace=replace)
                return parents[indexes]
        return crossover_func


class Mutation:
    """Mutation contains functions that return functions
       that mutate genes.
    """

    @staticmethod
    def _create_mutation_mask(mutation_rates, num_offspring):
        """Creates a mutation mask.
        params:
            mutation_rates: A list of floats within 0-1 (exclusive)
            num_offspring: An integer
        return: A numpy ndarray
        """
        x = []
        for rate in mutation_rates:
            x.append(
                np.random.binomial(1, rate,
                                   size=num_offspring)
            )
        return np.stack(x, axis=1)

    @classmethod
    def additive(cls, mutation_rates, distributions, normal=True,
                 round_values=False, variable_size=False):
        """Creates a mutation function that can add to the current value
           of a gene.
        params:
            mutation_rates: A list of floats within 0-1 (exclusive),
                            or a single float if variable size is True
            distributions: A list of either lower and upper bounds
                           or means and standard deviations
                           (depends on param normal), or a single
                           distribution if variable size is True
            normal: A boolean, which determines if the random distribution
                    is normal or uniform
            round_values: A boolean, which determines if mutations should be
                          rounded to the nearest whole integer
            variable_size: A boolean, which determines if the number of genes
                           in the genome can change
        return: A mutation function
        """
        if variable_size:
            if isinstance(mutation_rates, list) and len(mutation_rates) == 1:
                mutation_rate = mutation_rates[0]
            elif isinstance(mutation_rates, (int, float)):
                mutation_rate = mutation_rates
            else:
                raise ValueError('To have variable sizes there '
                                 'must only be one rate.')
            if (isinstance(distributions[0], (list, tuple))
                    or len(distributions) != 2):
                raise ValueError('To have variable sizes there '
                                 'must only be one distribution.')
            b, a = distributions

            def mutation_func(offspring):
                if normal:
                    for ndx in range(len(offspring)):
                        mask = np.random.binomial(1, mutation_rate,
                                                  size=offspring[ndx].shape)
                        mutation = np.random.normal(
                            b, a, size=offspring[ndx].shape
                        )
                        if round_values:
                            mutation = mutation.round()
                        offspring[ndx] = mask * mutation + offspring[ndx]
                else:
                    for ndx in range(len(offspring)):
                        mask = np.random.binomial(1, mutation_rate,
                                                  size=offspring[ndx].shape)
                        mutation = np.random.uniform(
                            b, a, size=offspring[ndx].shape
                        )
                        if round_values:
                            mutation = mutation.round()
                        offspring[ndx] = mask * mutation + offspring[ndx]
                return offspring
        else:
            distributions = np.asarray(distributions)
            b = distributions[:, 0]
            a = distributions[:, 1]

            def mutation_func(offspring):
                assert len(mutation_rates) == offspring.shape[1], (
                    'The number of mutation rates must match the '
                    'number of offspring.'
                )
                if normal:
                    m = np.random.normal(b, a, size=offspring.shape)
                else:
                    m = np.random.uniform(b, a, size=offspring.shape)
                if round_values:
                    m = m.round()
                mm = cls._create_mutation_mask(mutation_rates,
                                               offspring.shape[0])
                return m * mm + offspring
        return mutation_func

    @classmethod
    def variable(cls, mutation_rates, distributions, normal=True,
                 round_values=False, variable_size=False):
        """Creates a mutation function that can sets the value
           of a gene.
        params:
            mutation_rates: A list of floats within 0-1 (exclusive),
                            or a single float if variable size is True
            distributions: A list of either lower and upper bounds
                           or means and standard deviations
                           (depends on param normal), or a single
                           distribution if variable size is True
            normal: A boolean, which determines if the random distribution
                    is normal or uniform
            round_values: A boolean, which determines if mutations should be
                          rounded to the nearest whole integer
            variable_size: A boolean, which determines if the number of genes
                           in the genome can change
        return: A mutation function
        """
        if variable_size:
            if isinstance(mutation_rates, list) and len(mutation_rates) == 1:
                mutation_rate = mutation_rates[0]
            elif isinstance(mutation_rates, (int, float)):
                mutation_rate = mutation_rates
            else:
                raise ValueError('To have variable sizes there '
                                 'must only be one rate.')
            if (isinstance(distributions[0], (list, tuple))
                    or len(distributions) != 2):
                raise ValueError('To have variable sizes there '
                                 'must only be one distribution.')
            b, a = distributions

            def mutate_func(offspring):
                if normal:
                    for ndx in range(len(offspring)):
                        mask = np.random.binomial(1, mutation_rate,
                                                  size=offspring[ndx].shape)
                        if round_values:
                            mutation = np.random.normal(
                                b, a, size=offspring[ndx].shape
                            ).round()
                        else:
                            mutation = np.random.normal(
                                b, a, size=offspring[ndx].shape
                            )
                        offspring[ndx] = (mask * mutation + offspring[ndx] *
                                          (1 - mask))
                else:
                    for ndx in range(len(offspring)):
                        mask = np.random.binomial(1, mutation_rate,
                                                  size=offspring[ndx].shape)
                        if round_values:
                            mutation = np.random.uniform(
                                b, a, size=offspring[ndx].shape
                            ).round()
                        else:
                            mutation = np.random.uniform(
                                b, a, size=offspring[ndx].shape
                            )
                        offspring[ndx] = (mask * mutation + offspring[ndx] *
                                          (1 - mask))
                return offspring
        else:
            distributions = np.asarray(distributions)
            b = distributions[:, 0]
            a = distributions[:, 1]

            def mutate_func(offspring):
                assert len(mutation_rates) == offspring.shape[1], (
                    'The number of mutation rates must match the '
                    'number of offspring.'
                )
                if normal:
                    m = np.random.normal(b, a, size=offspring.shape)
                else:
                    m = np.random.uniform(b, a, size=offspring.shape)
                if round_values:
                    m = m.round()
                mm = cls._create_mutation_mask(mutation_rates,
                                               offspring.shape[0])
                mutated_offspring = m * mm + offspring * (1-mm)
                if round_values:
                    return mutated_offspring.round()
                return mutated_offspring
        return mutate_func


class SizeMutation:
    """SizeMutation contains functions that return functions
       that mutate the genome size.
    """

    @staticmethod
    def genome_double(value=None):
        """Creates a size mutation function that doubles the
           size of the current genome.
        params:
            value: A value to set the new genes to
                (Default copies current genome values)
        return: An incomplete size mutation function
        """
        if value is None:
            def size_mutation_partial_func(genome):
                assert genome.ndim == 1, (
                    'Genome must have 1 dimension.'
                )
                return np.hstack((genome, genome))
        else:
            def size_mutation_partial_func(genome):
                assert genome.ndim == 1, (
                    'Genome must have 1 dimension.'
                )
                return np.hstack((
                    genome,
                    np.full_like(genome, value)
                ))
        return size_mutation_partial_func

    @staticmethod
    def genome_half(keep_left=True):
        """Creates a size mutation function that halfs the
           size of the current genome.
        params:
            keep_left: A boolean, which determines if the
                       left or right size should be kept
        return: An incomplete size mutation function
        """
        def size_mutation_partial_func(genome):
            assert genome.ndim == 1, (
                'Genome must have 1 dimension.'
            )
            if len(genome) > 1:
                if keep_left:
                    return genome[:len(genome)//2]
                else:
                    return genome[len(genome)//2:]
            return genome

        return size_mutation_partial_func

    @staticmethod
    def random_gene_addition(value=None):
        """Creates a size mutation function that randomly
           inserts a gene in the genome.
        params:
            value: A value to set the new gene to
                   (Default copies current genome value)
        return: An incomplete size mutation function
        """
        if value is None:
            def size_mutation_partial_func(genome):
                assert genome.ndim == 1, (
                    'Genome must have 1 dimension.'
                )
                ndx = np.random.randint(1, len(genome)+1)
                return np.hstack((genome[:ndx],
                                  genome[ndx-1:]))
        else:
            def size_mutation_partial_func(genome):
                assert genome.ndim == 1, (
                    'Genome must have 1 dimension.'
                )
                ndx = np.random.randint(0, len(genome)+1)
                return np.hstack((genome[:ndx], value,
                                  genome[ndx:]))
        return size_mutation_partial_func

    @staticmethod
    def random_gene_deletion():
        """Creates a size mutation function that randomly
           deletes a gene in the genome.
        return: An incomplete size mutation function
        """
        def size_mutation_partial_func(genome):
            assert genome.ndim == 1, (
                'Genome must have 1 dimension.'
            )
            ndx = np.random.randint(0, len(genome))
            return np.hstack((genome[:ndx],
                              genome[ndx+1:]))

        return size_mutation_partial_func

    @staticmethod
    def first_gene_addition(value=None):
        """Creates a size mutation function that inserts
           a gene in the begining of the genome.
        params:
            value: A value to set the new gene to
                   (Default copies current genome value)
        return: An incomplete size mutation function
        """
        if value is None:
            def size_mutation_partial_func(genome):
                assert genome.ndim == 1, (
                    'Genome must have 1 dimension.'
                )
                return np.hstack((genome[0], genome))
        else:
            def size_mutation_partial_func(genome):
                assert genome.ndim == 1, (
                    'Genome must have 1 dimension.'
                )
                return np.hstack((value, genome))
        return size_mutation_partial_func

    @staticmethod
    def first_gene_deletion():
        """Creates a size mutation function that deletes
           a gene at the begining of the genome.
        return: An incomplete size mutation function
        """
        def size_mutation_partial_func(genome):
            assert genome.ndim == 1, (
                'Genome must have 1 dimension.'
            )
            return genome[1:]

        return size_mutation_partial_func

    @staticmethod
    def last_gene_addition(value=None):
        """Creates a size mutation function that inserts
           a gene at the end of the genome.
        params:
            value: A value to set the new gene to
                   (Default copies current genome value)
        return: An incomplete size mutation function
        """
        if value is None:
            def size_mutation_partial_func(genome):
                assert genome.ndim == 1, (
                    'Genome must have 1 dimension.'
                )
                return np.hstack((genome, genome[-1]))
        else:
            def size_mutation_partial_func(genome):
                assert genome.ndim == 1, (
                    'Genome must have 1 dimension.'
                )
                return np.hstack((genome, value))
        return size_mutation_partial_func

    @staticmethod
    def last_gene_deletion():
        """Creates a size mutation function that deletes
           a gene at the end of the genome.
        return: An incomplete size mutation function
        """
        def size_mutation_partial_func(genome):
            assert genome.ndim == 1, (
                'Genome must have 1 dimension.'
            )
            return genome[:-1]

        return size_mutation_partial_func

    @staticmethod
    def complete_mutations(size_mutation_rate, probabilities, funcs):
        """Creates a complete size mutation function from incomplete
           size mutation functions.
        params:
            size_mutation_rate: A float within 0-1 (exclusive), which is the
                                rate of a genome size mutating
            probabilities: A list of floats within 0-1 (exclusive), which
                           contains the chance of each size mutation function
                           being used
            funcs: A list of incomplete size mutation functions
        return: A complete size mutation function
        """
        arange = np.arange(len(funcs))

        def size_mutation_func(offspring):
            new_offspring = []
            for genome in offspring:
                if np.random.random() < size_mutation_rate:
                    choice = np.random.choice(arange, p=probabilities)
                    new_offspring.append(funcs[choice](genome))
                else:
                    new_offspring.append(genome)
            return new_offspring

        return size_mutation_func


class EvolutionAlgorithm:
    """EvolutionAlgorithm is a class that is able to simulate 'natural'
       selection of genes and genomes.
    """

    def __init__(self, fitness_func, selection_func, mutation_func,
                 crossover_func, size_mutation_func=None):
        """Creates an evolution algorithm by the provided functions.
        params:
            fitness_func: A function that takes a list or numpy ndarray of
                          genomes (offspring), and returns list of fitness
                          values
            selection_func: A function that takes a list or numpy ndarray of
                            genomes (offspring) and fitness values, and
                            returns the selected genomes (offspring)
            mutation_func: A function that takes a list or numpy ndarray of
                           genomes (offspring), and returns the offspring
                           mutated
            crossover_func: A function that takes a list or numpy array of
                            genomes (parents), and returns offspring
            size_mutation_func: A function that takes a list or numpy
                                ndarray of genomes, and returns the
                                genomes with mutated sizes
        """
        self.get_fitness = fitness_func
        self.select = selection_func
        self.crossover = crossover_func
        self.mutate = mutation_func
        self.mutate_size = size_mutation_func

    def simulate(self, base_genome, generations, population, selection_size,
                 return_all_genomes=False, verbose=True):
        """Simulates natural selection of genomes.
        params:
            base_genome: A list of floats or integers (genes)
            generations: An integer, which is the number of complete cycles of
                         performing crossovers, mutations, and selections on
                         the entire population
            population: An integer, which is the number of genomes in a
                        generation
            selection_size: An integer, which is the number of
                            offspring to select from the population
                            each generation/cycle
            return_all_genomes: A boolean, which determiens if all
                                the genomes with their corresponding
                                fitness values should be returned
            verbose: A boolean, which determines if information
                     will be printed to the console
        return: A list of tuples each containing a fitness value and a genome
        """
        assert population > selection_size, (
            'The population must be greater than the selection size.'
        )
        all_genomes = []
        parents = np.array([base_genome])
        for generation in range(generations):
            offspring = self.crossover(parents, population)
            if self.mutate_size is not None:
                offspring = self.mutate_size(offspring)
            offspring = self.mutate(offspring)
            fitness_values = self.get_fitness(offspring)
            if return_all_genomes:
                all_genomes += list(zip(fitness_values, parents))
            selected_offspring = self.select(offspring, fitness_values,
                                             selection_size)
            parents = selected_offspring
            if verbose:
                highest = fitness_values.max()
                lowest = fitness_values.min()
                mean = fitness_values.mean()
                print(f'Generation {generation+1}\n'
                      f'Highest Fitness: {highest} - '
                      f'Lowest Fitness: {lowest} - '
                      f'Mean Fitness: {mean}')
        if return_all_genomes:
            return all_genomes
        return list(zip(self.get_fitness(parents), parents))

    def simulate_islands(self, base_genome, generations, population,
                         selection_size, islands, island_migrations,
                         threaded=False, verbose=True):
        """Simulates natural selection of genomes with isolating islands.
        params:
            base_genome: A list of floats or integers (genes)
            generations: An integer, which is the number of complete cycles of
                         performing crossovers, mutations, and selections on
                         the entire population
            population: An integer, which is the number of genomes in a
                        generation
            selection_size: An integer, which is the number of
                            offspring to select from the population
                            each generation/cycle
            islands: An integer, which is the number of isolated islands in
                     the simulation
            island_migrations: An integer, which is the number of migrations
                               of the offspring between the isolated islands
            threaded: A boolean, which determines if the islands should be run
                      on in parallel
            verbose: A boolean, which determines if information
                     will be printed to the console
        return: A list of tuples each containing a fitness value and a genome
        """
        assert population > selection_size, (
            'The population must be greater than the selection size.'
        )
        assert generations > island_migrations, (
            'Generations must be greater than island migrations.'
        )
        assert islands > 1 or island_migrations == 1, (
            'Island migrations should be one if there is only one island.'
        )

        def island(params):
            island_num, parents, start, end = params
            for generation in range(start, end):
                offspring = self.crossover(parents, population)
                if self.mutate_size is not None:
                    offspring = self.mutate_size(offspring)
                offspring = self.mutate(offspring)
                fitness_values = self.get_fitness(offspring)
                selected_offspring = self.select(offspring, fitness_values,
                                                 selection_size)
                parents = selected_offspring
                if verbose:
                    highest = fitness_values.max()
                    lowest = fitness_values.min()
                    mean = fitness_values.mean()
                    print(f'Island({island_num+1}) '
                          f'Generation {generation+1}\n'
                          f'Highest Fitness: {highest} - '
                          f'Lowest Fitness: {lowest} - '
                          f'Mean Fitness: {mean}\n', end='')
            return parents

        pool = ThreadPool(islands)
        generations = round(generations / island_migrations)
        parents = np.array([base_genome])
        for ndx in range(island_migrations):
            if verbose:
                print(f'\nMigration {ndx+1}')
            all_parents = []
            if threaded:
                params = []
                for ndx2 in range(islands):
                    params.append((ndx2, parents,
                                   generations * ndx,
                                   generations * (ndx + 1)))
                results = pool.map(island, params)
                for result in results:
                    if isinstance(result, np.ndarray):
                        all_parents.append(result)
                    else:
                        all_parents += result
            else:
                for ndx2 in range(islands):
                    parents = island((ndx2, parents,
                                      generations * ndx,
                                      generations * (ndx + 1)))
                    if isinstance(parents, np.ndarray):
                        all_parents.append(parents)
                    else:
                        all_parents += parents
            if isinstance(all_parents[0], np.ndarray):
                all_parents = np.vstack(all_parents)
            parents = all_parents

        return list(zip(self.get_fitness(parents), parents))


class HyperparameterTuner:
    """This class is used for tuning hyper parameters."""

    def __init__(self):
        """Initalizes lists to keep track of parameters.
        """
        self.num_parameters = 0
        self.mutation_distributions = []
        self.mutation_volatilities = []
        self.initial_values = []
        self.parameters = []
        self.tuning = False

    def tune(self, generations, population, selection_size,
             eval_func, lowest_best=True, crossover_func=None,
             verbose=False):
        """Tunes the parameters to get the best parameters with
           an evolution algorithim.
        params:
            generations: An integer, which is the number of complete cycles of
                         performing crossovers, mutations, and selections on
                         the entire population
            population: An integer, which is the number of genomes in a
                        generation
            selection_size: An integer, which is the number of
                            parameter combinations to select from the
                            population each generation/cycle
            eval_func: A function, which returns a single value that
                       represents the parameters fitness
            lowest_best: A boolean, which determines if lower fitness values
                         are better or worse
            crossover_func: A function that takes a list or numpy array of
                            genomes (parents), and returns offspring
                            (defaults to no crossover)
            verbose: A boolean, which determines if the evolution
                     algorithm should print information to the screen
        """
        self.tuning = True

        if lowest_best:
            selection = Selection.select_lowest()
        else:
            selection = Selection.select_highest()
        if crossover_func is None:
            crossover_func = Crossover.single()

        def fitness_func(offspring):
            errors = []
            for genome in offspring:
                self.parameters = genome
                errors.append(eval_func())
            return np.array(errors)

        ea = EvolutionAlgorithm(
            fitness_func,
            selection,
            Mutation.variable(self.mutation_volatilities,
                              self.mutation_distributions,
                              normal=False),
            crossover_func
        )
        genomes = ea.simulate(self.initial_values, generations,
                              population, selection_size, False,
                              verbose=verbose)
        genomes = sorted(genomes, key=lambda x: x[0])
        self.tuning = False
        if lowest_best:
            self.parameters = genomes[0][1]
        self.parameters = genomes[-1][1]

    def uniform(self, lower_bound, upper_bound, volatility=.1,
                inital_value=None, integer=False):
        """Returns a function that when called returns the
           value of that parameter.
        params:
            lower_bound: A float or integer, which is the lowest
                         value that the parameter can be mutated to
            upper_bound: A float or integer, which is the highest
                         value that the parameter can be mutated to
            volatility: A float, which is the rate that this parameter
                        is mutated
            inital_value: A float or integer, which is the starting value
                          of the parameter
            integer: A boolean, which determiens if the parameter should
                      be rounded and cast to an integer
        return: A parameter function, which returns a number in the
                uniform range
        """
        if self.tuning:
            raise Exception('Parameters cannot be added while tuning')
        self.mutation_distributions.append([lower_bound, upper_bound])
        self.mutation_volatilities.append(volatility)
        if inital_value is None:
            self.initial_values.append((lower_bound + upper_bound) / 2)
        else:
            self.initial_values.append(inital_value)
        self.parameters.append(self.initial_values[-1])
        ndx = self.num_parameters

        def parameter():
            nonlocal ndx
            if integer:
                return int(round(self.parameters[ndx]))
            return self.parameters[ndx]

        self.num_parameters += 1
        return parameter

    def list(self, alist, volatility=.1,
             inital_ndx=None):
        """Returns a function that when called returns a element
           from the list.
        params:
            alist: A list of values, which can be mutated to
            volatility: A float, which is the rate that this parameter
                        is mutated
            inital_ndx: A integer, which is the starting index
                        of the parameter
        return: A parameter function, which returns a number in the
                uniform range
        """
        if self.tuning:
            raise Exception('Parameters cannot be added while tuning')
        self.mutation_distributions.append([0, len(alist)-1])
        self.mutation_volatilities.append(volatility)
        if inital_ndx is None:
            self.initial_values.append(len(alist) // 2)
        else:
            self.initial_values.append(inital_ndx)
        self.parameters.append(self.initial_values[-1])
        ndx = self.num_parameters

        def parameter():
            nonlocal ndx
            return alist[int(round(self.parameters[ndx]))]

        self.num_parameters += 1
        return parameter

    def boolean(self, volatility=.1, inital_value=True):
        """Returns a function that when called returns the
           value of that parameter.
        params:
            volatility: A float, which is the rate that this parameter
                        is mutated
            inital_value: A boolean, which is the starting value
                          of the parameter
        return: A parameter function, which returns a boolean
        """
        if self.tuning:
            raise Exception('Parameters cannot be added while tuning')
        self.mutation_distributions.append([0, 2])
        self.mutation_volatilities.append(volatility)
        self.initial_values.append(1 if inital_value else 0)
        self.parameters.append(self.initial_values[-1])
        ndx = self.num_parameters

        def parameter():
            nonlocal ndx
            return self.parameters[ndx] == 1

        self.num_parameters += 1
        return parameter
