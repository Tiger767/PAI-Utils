"""
Author: Travis Hammond
Version: 12_9_2020
"""

import pytest

from paiutils.evolution_algorithm import *


def test_fitness():
    ff = Fitness.match_mse([.2, 1, 5, 10])
    assert ff(np.array([[.2, 1, 5, 10]]))[0] - 0 < .000000001
    assert ff(np.array([[.2, 1, 5, 15]]))[0] - 6.25 < .000000001
    assert ff(np.array([[.3, 1, 5, 10]]))[0] - 0.0025 < .000000001

    ff = Fitness.match_mse([.2, 1, 5, 10], variable_size=True)
    assert ff(np.array([[.2, 1, 5, 10]]))[0] - 0 < .000000001
    assert ff(np.array([[.2, 1, 5, 15]]))[0] - 6.25 < .000000001
    assert ff(np.array([[.3, 1, 5, 10]]))[0] - 0.0025 < .000000001
    assert ff(np.array([[.2, 1, 5, 10, 1]]))[0] - 1 < .000000001
    assert ff(np.array([[.2, 1, 5, 10, 1, 5]]))[0] - 4 < .000000001
    assert ff(np.array([[.2]]))[0] - 9 < .000000001


def test_selection():
    sf = Selection.select_highest()
    assert (sf(np.array([[.2, 1, 5, 10], [5, 4, 3, 2]]), np.array(
        [5, 3]), 1)[0] == np.array([.2, 1, 5, 10])).all()
    assert (sf(np.array([[.2, 1, 5, 10], [5, 4, 3, 2]]), np.array(
        [3, 5]), 1)[0] == np.array([5, 4, 3, 2])).all()
    assert (sf(np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [9, 4, 3, 2]]), np.array(
        [5, 3, 9]), 2) == np.array([[.2, 1, 5, 10], [9, 4, 3, 2]])).all()

    sf = Selection.select_highest(variable_size=True)
    assert sf([[.2, 1, 5], [5, 4, 3, 2]], np.array([5, 3]), 1)[0] == [.2, 1, 5]
    assert sf([[.2, 1, 5, 10], [5, 2]], np.array([3, 5]), 1)[0] == [5, 2]
    assert sf([[.2, 1, 5, 10, 9, 8], [5, 4, 3], [9, 5, 4, 3, 2]], np.array(
        [5, 3, 9]), 2) == [[.2, 1, 5, 10, 9, 8], [9, 5, 4, 3, 2]]

    sf = Selection.select_lowest()
    assert (sf(np.array([[.2, 1, 5, 10], [5, 4, 3, 2]]), np.array(
        [5, 3]), 1)[0] == np.array([5, 4, 3, 2])).all()
    assert (sf(np.array([[.2, 1, 5, 10], [5, 4, 3, 2]]), np.array(
        [3, 5]), 1)[0] == np.array([.2, 1, 5, 10])).all()
    assert (sf(np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [9, 4, 3, 2]]), np.array(
        [5, 3, 9]), 2) == np.array([[5, 4, 3, 2], [.2, 1, 5, 10]])).all()

    sf = Selection.select_lowest(variable_size=True)
    assert sf([[.2, 1, 5], [5, 4, 3, 2]], np.array([5, 3]), 1)[
        0] == [5, 4, 3, 2]
    assert sf([[.2, 1, 5, 10], [5, 2]], np.array([3, 5]), 1)[
        0] == [.2, 1, 5, 10]
    assert sf([[.2, 1, 5, 10, 9, 8], [5, 4, 3], [9, 5, 4, 3, 2]],
              np.array([5, 3, 9]), 2) == [[5, 4, 3], [.2, 1, 5, 10, 9, 8]]


def test_crossover():
    cf = Crossover.dual()
    assert cf(np.array([[.2, 1, 5, 10], [5, 4, 3, 2],
                        [1, 1, 2, 2]]), 10).shape == (10, 4)
    assert cf(np.array([[.2, 1, 5, 10]]), 10).shape == (10, 4)

    cf = Crossover.triple()
    assert cf(np.array([[.2, 1, 5, 10], [5, 4, 3, 2],
                        [1, 1, 2, 2]]), 10).shape == (10, 4)
    assert cf(np.array([[.2, 1, 5, 10]]), 10).shape == (10, 4)

    cf = Crossover.population_avg()
    assert cf(np.array([[.2, 1, 5, 10], [5, 4, 3, 2],
                        [1, 1, 2, 2]]), 5).shape == (5, 4)
    assert cf(np.array([[.2, 1, 5, 10]]), 5).shape == (5, 4)

    cf = Crossover.population_shuffle()
    assert cf(np.array([[.2, 1, 5, 10], [5, 4, 3, 2],
                        [1, 1, 2, 2]]), 5).shape == (5, 4)
    assert cf(np.array([[.2, 1, 5, 10]]), 5).shape == (5, 4)

    cf = Crossover.single()
    assert cf(np.array([[.2, 1, 5, 10], [5, 4, 3, 2],
                        [1, 1, 2, 2]]), 5).shape == (5, 4)
    assert cf(np.array([[.2, 1, 5, 10]]), 5).shape == (5, 4)

    cf = Crossover.single(variable_size=True)
    assert len(
        cf(np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [1, 1, 2, 2]]), 5)) == 5
    assert len(cf(np.array([[.2, 1, 5, 10]]), 5)) == 5


def test_mutation():
    mf = Mutation.additive([.5, .3, .2, .1], [(0, .1), (1, 5), (-1, 1), (-10, 10)],
                           normal=True)
    assert mf(
        np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [1, 1, 2, 2]])).shape == (3, 4)

    mf = Mutation.additive([.5, .3, .2, .1], [(0, .1), (1, 5), (-1, 1), (-10, 10)],
                           normal=False)
    assert mf(
        np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [1, 1, 2, 2]])).shape == (3, 4)

    mf = Mutation.additive([.5, .3, .2, .1], [(0, .1), (1, 5), (-1, 1), (-10, 10)],
                           normal=True, round_values=True)
    assert mf(
        np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [1, 1, 2, 2]])).shape == (3, 4)

    mf = Mutation.additive(.5, (-5, 5),
                           variable_size=True)
    assert mf(
        np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [1, 1, 2, 2]])).shape == (3, 4)

    mf = Mutation.additive(.5, (-5, 5),
                           normal=False, variable_size=True)
    assert mf(
        np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [1, 1, 2, 2]])).shape == (3, 4)

    with pytest.raises(ValueError):
        mf = Mutation.additive([.5, .3, .2, .1], [(0, .1), (1, 5), (-1, 1), (-10, 10)],
                               normal=False, variable_size=True)
        assert mf(
            np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [1, 1, 2, 2]])).shape == (3, 4)

    mf = Mutation.variable([.5, .3, .2, .1], [(0, .1), (1, 5), (-1, 1), (-10, 10)],
                           normal=True)
    assert mf(
        np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [1, 1, 2, 2]])).shape == (3, 4)

    mf = Mutation.variable([.5, .3, .2, .1], [(0, .1), (1, 5), (-1, 1), (-10, 10)],
                           normal=False)
    assert mf(
        np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [1, 1, 2, 2]])).shape == (3, 4)

    mf = Mutation.variable([.5, .3, .2, .1], [(0, .1), (1, 5), (-1, 1), (-10, 10)],
                           normal=True, round_values=True)
    assert mf(
        np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [1, 1, 2, 2]])).shape == (3, 4)

    mf = Mutation.variable(.5, (-5, 5),
                           variable_size=True)
    assert mf(
        np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [1, 1, 2, 2]])).shape == (3, 4)

    mf = Mutation.variable(.5, (-5, 5),
                           normal=False, variable_size=True)
    assert mf(
        np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [1, 1, 2, 2]])).shape == (3, 4)

    with pytest.raises(ValueError):
        mf = Mutation.variable([.5, .3, .2, .1], [(0, .1), (1, 5), (-1, 1), (-10, 10)],
                               normal=False, variable_size=True)
        assert mf(
            np.array([[.2, 1, 5, 10], [5, 4, 3, 2], [1, 1, 2, 2]])).shape == (3, 4)


def test_size_mutation():
    smf = SizeMutation.genome_double()
    assert len(smf(np.array([.2, 1, 5, 10]))) == 8
    assert (smf(np.array([.2, 1])) == np.array([.2, 1, .2, 1])).all()

    smf = SizeMutation.genome_double(value=5)
    assert len(smf(np.array([.2, 1, 5, 10]))) == 8
    assert (smf(np.array([.2, 1])) == np.array([.2, 1, 5, 5])).all()

    smf = SizeMutation.genome_half()
    assert len(smf(np.array([.2, 1, 5, 10]))) == 2
    assert (smf(np.array([.2, 1])) == np.array([.2])).all()

    smf = SizeMutation.genome_half(keep_left=False)
    assert len(smf(np.array([.2, 1, 5, 10]))) == 2
    assert (smf(np.array([.2, 1])) == np.array([1])).all()

    smf = SizeMutation.random_gene_addition()
    assert len(smf(np.array([.2, 1, 5, 10]))) == 5

    smf = SizeMutation.random_gene_addition(value=5)
    assert len(smf(np.array([.2, 1, 5, 10]))) == 5

    smf = SizeMutation.random_gene_deletion()
    assert len(smf(np.array([.2, 1, 5, 10]))) == 3

    smf = SizeMutation.first_gene_addition()
    assert len(smf(np.array([.2, 1, 5, 10]))) == 5

    smf = SizeMutation.first_gene_addition(value=5)
    assert len(smf(np.array([.2, 1, 5, 10]))) == 5

    smf = SizeMutation.first_gene_deletion()
    assert len(smf(np.array([.2, 1, 5, 10]))) == 3

    smf = SizeMutation.last_gene_addition()
    assert len(smf(np.array([.2, 1, 5, 10]))) == 5

    smf = SizeMutation.last_gene_addition(value=5)
    assert len(smf(np.array([.2, 1, 5, 10]))) == 5

    smf = SizeMutation.last_gene_deletion()
    assert len(smf(np.array([.2, 1, 5, 10]))) == 3

    smf = SizeMutation.complete_mutations(
        1, [.5, .5],
        [SizeMutation.random_gene_deletion(),
         SizeMutation.random_gene_addition()]
    )
    length = len(smf([np.array([.2, 1, 5, 10])])[0])
    assert length == 3 or length == 5


def test_evolution_algorithm():
    ff = Fitness.match_mse(np.array([5, 4, 9, 1]))
    sf = Selection.select_lowest()
    mf = Mutation.additive(
        [.2, .2, .2, .2], [(0, 1), (0, 1), (0, 1), (0, 1)]
    )
    cf = Crossover.population_shuffle()
    evoalgo = EvolutionAlgorithm(ff, sf, mf, cf)
    results = evoalgo.simulate([0, 0, 0, 0], 100, 100, 3)
    assert results[0][0] < .0005

    results = evoalgo.simulate_islands([0, 0, 0, 0], 10, 100, 3, 10, 5)
    assert results[0][0] < .0005

    ff = Fitness.match_mse(
        np.array([.5, .4, .9, .1, .9, .3]), variable_size=True)
    sf = Selection.select_lowest(variable_size=True)
    mf = Mutation.additive(.2, (0, 1), variable_size=True)
    cf = Crossover.single(variable_size=True)
    smf = SizeMutation.complete_mutations(
        .5, [.5, .5],
        [SizeMutation.random_gene_deletion(),
         SizeMutation.random_gene_addition(value=0)]
    )
    evoalgo = EvolutionAlgorithm(ff, sf, mf, cf, size_mutation_func=smf)
    results = evoalgo.simulate([0, 0, 0, 0], 100, 100, 3)
    assert results[0][0] < .0005

    results = evoalgo.simulate_islands([0, 0, 0, 0], 10, 100, 3, 10, 5)
    assert results[0][0] < .0005


def test_hyperparameter_tuner():
    hpt = HyperparameterTuner()
    a = hpt.uniform(0, 10)
    b = hpt.uniform(0, 5)
    c = hpt.boolean()
    d = hpt.list([1, 2, 3, 1])
    e = hpt.list([-1, 1])

    def eval_func():
        if c():
            return a() * b() - 10
        else:
            if a() > 8:
                return (a() * b())**d() * e()
            return (a() * b())**d()

    hpt.tune(100, 100, 3, eval_func, verbose=True)
    print(a(), b(), c())
    assert eval_func() < -8
