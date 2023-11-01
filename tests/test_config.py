import pytest
from hakowan import config

class TestConfig:
    def test_default(self):
        cfg = config()
        assert len(cfg.emitters) > 0
        assert cfg.emitters[0].filename.exists()
