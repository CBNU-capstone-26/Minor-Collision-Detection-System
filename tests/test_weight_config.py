import importlib


def test_weight_path_uses_environment_override(monkeypatch):
    monkeypatch.setenv("HITANDRUN_WEIGHTS_PATH", "/tmp/test-s3d.pth")
    import model.config as config

    importlib.reload(config)

    assert config.SERVICE_WEIGHTS_PATH.name == "test-s3d.pth"
    assert config.PREDICT_WEIGHTS_PATH == config.SERVICE_WEIGHTS_PATH
    assert config.EVAL_WEIGHTS_PATH == config.SERVICE_WEIGHTS_PATH
