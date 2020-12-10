"""
Author: Travis Hammond
Version: 12_9_2020
"""


from paiutils.analytics import *


def test_calculate_distribution_of_labels():
    a = Analyzer(
        np.random.random(size=(100, 10)),
        np.random.randint(0, 10, size=(100)),
        labels=np.arange(10).tolist()
    )
    assert len(a.calculate_distribution_of_labels()) == 10
    assert sum(a.calculate_distribution_of_labels().values()) == 100

def test_create_label_ndx_groups():
    a = Analyzer(
        np.random.random(size=(1000, 10)),
        np.random.randint(0, 5, size=(1000)),
        labels=np.arange(5).tolist()
    )
    assert len(a.create_label_ndx_groups().keys()) == 5
    assert sum([len(x) for x in a.create_label_ndx_groups().values()]) == 1000

def test_shrink_data():
    a = Analyzer(
        np.random.random(size=(1000, 10)),
        np.random.randint(0, 5, size=(1000)),
        labels=np.arange(5).tolist()
    )
    assert sum(a.calculate_distribution_of_labels().values()) == 1000
    a = a.shrink_data(5)
    assert sum(a.calculate_distribution_of_labels().values()) == 25

    a = Analyzer(
        np.random.random(size=(1000, 10)),
        np.random.randint(0, 5, size=(1000)),
        labels=np.arange(5).tolist()
    )
    pre_dl = a.calculate_distribution_of_labels()
    a = a.shrink_data({0: 1, 1: 3, 2: 5})
    dl = a.calculate_distribution_of_labels()
    assert dl[0] == 1
    assert dl[1] == 3
    assert dl[2] == 5
    assert dl[3] == pre_dl[3]
    assert dl[4] == pre_dl[4]

    a.shrink_data({1: 1})
    dl = a.calculate_distribution_of_labels()
    assert dl[1] == 3

def test_expand_data():
    a = Analyzer(
        np.random.random(size=(100, 10)),
        np.random.randint(0, 5, size=(100)),
        labels=np.arange(5).tolist()
    )
    assert sum(a.calculate_distribution_of_labels().values()) == 100
    a = a.expand_data(100)
    assert sum(a.calculate_distribution_of_labels().values()) == 500
    a = a.expand_data(50)  # should not do anything
    assert sum(a.calculate_distribution_of_labels().values()) == 500
    a = a.expand_data({0: 10, 1: 10})  # should not do anything
    assert sum(a.calculate_distribution_of_labels().values()) == 500

    a = Analyzer(
        np.random.random(size=(1000, 10)),
        np.random.randint(0, 5, size=(1000)),
        labels=np.arange(5).tolist()
    )
    pre_dl = a.calculate_distribution_of_labels()
    a = a.expand_data({0: 500, 1: 505, 2: 400})
    dl = a.calculate_distribution_of_labels()
    assert dl[0] == 500
    assert dl[1] == 505
    assert dl[2] == 400
    assert dl[3] == pre_dl[3]
    assert dl[4] == pre_dl[4]

    a.expand_data({1: 600})
    dl = a.calculate_distribution_of_labels()
    assert dl[1] == 505

def test_plot():
    pass

def test_boxplot():
    pass

def test_isomap():
    a = Analyzer(
        np.random.random(size=(100, 10)),
        np.random.randint(0, 5, size=(100)),
        labels=np.arange(5).tolist()
    )
    assert a.isomap(n_components=1).shape == (100, 1)
    assert a.isomap(n_components=2).shape == (100, 2)
    assert a.isomap(n_components=3).shape == (100, 3)

def test_locally_linear_embedding():
    a = Analyzer(
        np.random.random(size=(100, 10)),
        np.random.randint(0, 5, size=(100)),
        labels=np.arange(5).tolist()
    )
    assert a.locally_linear_embedding(n_components=1).shape == (100, 1)
    assert a.locally_linear_embedding(n_components=2).shape == (100, 2)
    assert a.locally_linear_embedding(n_components=3).shape == (100, 3)

def test_mds():
    a = Analyzer(
        np.random.random(size=(100, 10)),
        np.random.randint(0, 5, size=(100)),
        labels=np.arange(5).tolist()
    )
    assert a.mds(n_components=1).shape == (100, 1)
    assert a.mds(n_components=2).shape == (100, 2)
    assert a.mds(n_components=3).shape == (100, 3)

def test_tsne():
    a = Analyzer(
        np.random.random(size=(100, 10)),
        np.random.randint(0, 5, size=(100)),
        labels=np.arange(5).tolist()
    )
    assert a.tsne(n_components=1).shape == (100, 1)
    assert a.tsne(n_components=2).shape == (100, 2)
    assert a.tsne(n_components=3).shape == (100, 3)
