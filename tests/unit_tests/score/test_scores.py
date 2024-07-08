import bnlearn as bn
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
import sempler

from dodiscover.score.score_function import ScoreFunction


def titanic_data():
    df = pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]})
    m1 = BayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
    m2 = BayesianNetwork([("C", "A"), ("C", "B"), ("A", "D")])

    # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
    titanic_data = bn.import_example(data="titanic")
    titanic_data = titanic_data[["Survived", "Sex", "Pclass"]]


class TestBDeuScore:
    def test_score(self):
        ScoreFunction()
        self.assertAlmostEqual(BDeuScore(self.d1).score(self.m1), -9.907103407446435)
        self.assertEqual(BDeuScore(self.d1).score(BayesianNetwork()), 0)

    def test_score_titanic(self):
        scorer = BDeuScore(self.titanic_data2, equivalent_sample_size=25)
        titanic = BayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        self.assertAlmostEqual(scorer.score(titanic), -1892.7383393910427)
        titanic2 = BayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        self.assertLess(scorer.score(titanic2), scorer.score(titanic))


class TestBDsScore(unittest.TestCase):
    def setUp(self):
        """Example taken from https://arxiv.org/pdf/1708.00689.pdf"""
        self.d1 = pd.DataFrame(
            data={
                "X": [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                "Y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                "Z": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                "W": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            }
        )
        self.m1 = BayesianNetwork([("W", "X"), ("Z", "X")])
        self.m1.add_node("Y")
        self.m2 = BayesianNetwork([("W", "X"), ("Z", "X"), ("Y", "X")])

    def test_score(self):
        self.assertAlmostEqual(
            BDsScore(self.d1, equivalent_sample_size=1).score(self.m1),
            -36.82311976667139,
        )
        self.assertEqual(
            BDsScore(self.d1, equivalent_sample_size=1).score(self.m2),
            -45.788991276221964,
        )

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2


class TestBicScore(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(
            data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
        )
        self.m1 = BayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
        self.m2 = BayesianNetwork([("C", "A"), ("C", "B"), ("A", "D")])

        # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv")
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_score(self):
        self.assertAlmostEqual(BicScore(self.d1).score(self.m1), -10.698440814229318)
        self.assertEqual(BicScore(self.d1).score(BayesianNetwork()), 0)

    def test_score_titanic(self):
        scorer = BicScore(self.titanic_data2)
        titanic = BayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        self.assertAlmostEqual(scorer.score(titanic), -1896.7250012840179)
        titanic2 = BayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        self.assertLess(scorer.score(titanic2), scorer.score(titanic))

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2
        del self.titanic_data
        del self.titanic_data2


class TestK2Score(unittest.TestCase):
    def setUp(self):
        self.d1 = pd.DataFrame(
            data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
        )
        self.m1 = BayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
        self.m2 = BayesianNetwork([("C", "A"), ("C", "B"), ("A", "D")])

        # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
        self.titanic_data = pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv")
        self.titanic_data2 = self.titanic_data[["Survived", "Sex", "Pclass"]]

    def test_score(self):
        self.assertAlmostEqual(K2Score(self.d1).score(self.m1), -10.73813429536977)
        self.assertEqual(K2Score(self.d1).score(BayesianNetwork()), 0)

    def test_score_titanic(self):
        scorer = K2Score(self.titanic_data2)
        titanic = BayesianNetwork([("Sex", "Survived"), ("Pclass", "Survived")])
        self.assertAlmostEqual(scorer.score(titanic), -1891.0630673606006)
        titanic2 = BayesianNetwork([("Pclass", "Sex")])
        titanic2.add_nodes_from(["Sex", "Survived", "Pclass"])
        self.assertLess(scorer.score(titanic2), scorer.score(titanic))

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2
        del self.titanic_data
        del self.titanic_data2


seed = 1234
rng = np.random.default_rng(seed)


def sample_data(n, p, scm):

    true_A = np.array(
        [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
    )
    factorization = [(4, (2, 3)), (3, (2,)), (2, (0, 1)), (0, ()), (1, ())]
    true_B = true_A * rng.uniform(1, 2, size=true_A.shape)
    scm = sempler.LGANM(true_B, (0, 0), (0.3, 0.4))
    n_features = len(true_A)
    n_samples = 10_000
    obs_data = scm.sample(n=n_samples)
    return true_A, obs_data


def test_bic_score_with_true_graph_is_better_than_empty():
    scorer = ScoreFunction(score="bic")
    true_score = scorer.full_score(true_A)

    empty_score = scorer.full_score(np.zeros_like(true_A))
    assert true_score > empty_score


def test_bic_score_preserves_decomposability():
    pass
