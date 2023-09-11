import pytest
from hakowan.beta import config

class TestConfig:
    def test_default(self):
        cfg = config.Config()
        assert len(cfg.emitters) > 0
        assert cfg.emitters[0].filename.exists()
