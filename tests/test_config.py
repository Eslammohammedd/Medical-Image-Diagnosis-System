from pathlib import Path

import yaml

from utils import config


def test_class_names_and_count():
    assert len(config.CLASS_NAMES) == config.NUM_CLASSES
    assert "Normal" in config.CLASS_NAMES
    assert "Covid-19" in config.CLASS_NAMES


def test_confidence_thresholds_are_ordered():
    assert 0 < config.CONFIDENCE_MEDIUM < config.CONFIDENCE_HIGH < 1


def test_weights_path_is_inside_models():
    assert config.BEST_WEIGHTS_PATH.parent == config.MODEL_WEIGHTS
    assert config.BEST_WEIGHTS_PATH.name == "best_model.pth"


def test_directories_created_on_import():
    assert config.MODEL_WEIGHTS.is_dir()
    assert config.LOG_DIR.is_dir()


def test_env_example_documents_required_keys():
    env_example = Path(config.PROJECT_ROOT) / ".env.example"
    content = env_example.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=" in content
    assert "ROBOFLOW_API_KEY=" in content
    assert "MODEL_PATH=" in content


def test_config_files_are_valid_yaml():
    configs_dir = Path(config.PROJECT_ROOT) / "configs"
    files = sorted(configs_dir.glob("*.yaml"))
    assert files
    for path in files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
