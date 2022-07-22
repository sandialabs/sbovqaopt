'''
Tests for sbovqaopt.optimizer module.
'''
from sbovqaopt.optimizer import Optimizer

import numpy as np


class TestOptimizer:

    def test_optimizer(self) -> None:
        optimizer = Optimizer()
        assert isinstance(optimizer.get_support_level(), dict)

        def objective(x: np.ndarray) -> float:
            return np.linalg.norm(x)
        x0 = np.array([0.05, 0.04, 0.03, 0.02])

        # test with nfev_final_avg=20, verify res.x and res.fun
        optimizer = Optimizer(patch_size=0.01, nfev_final_avg=20)
        res = optimizer.minimize(objective, x0)
        assert np.all(np.isclose(res.x, np.zeros_like(x0), atol=0.01))
        assert np.isclose(res.fun, 0, atol=0.01)

        # test with nfev_final_avg=0, verify that no res.fun is returned
        optimizer = Optimizer(patch_size=0.01, nfev_final_avg=0)
        res = optimizer.minimize(objective, x0)
        assert isinstance(res.fun, str)
