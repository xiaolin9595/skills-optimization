import os
import csv
import json
import pytest
from unittest.mock import MagicMock
from skill_opt.optimizer.greater import GreaterOptimizer
from skill_opt.core.config import AppConfig, OptimizeConfig

@pytest.fixture
def mock_optimizer():
    config = AppConfig(model_name="mock-model")
    with os.popen("touch mock_file") as f:
        pass
    
    # Mock load_model_and_tokenizer to avoid loading real models
    import skill_opt.optimizer.greater
    skill_opt.optimizer.greater.load_model_and_tokenizer = MagicMock(return_value=(MagicMock(), MagicMock()))
    
    optimizer = GreaterOptimizer(config)
    return optimizer

def test_load_bbh_csv(mock_optimizer, tmp_path):
    # Create a dummy BBH CSV file
    bbh_file = tmp_path / "bbh_test.json"
    content = [
        ["goal", "target", "final_target"],
        ["What is 1+1?", "2", "2"],
        ["Is the sun hot?", "Yes", "Yes"]
    ]
    with open(bbh_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(content)
    
    config = OptimizeConfig(
        dataset_path=str(bbh_file),
        num_examples=10
    )
    
    data = mock_optimizer._load_training_data_raw(config)
    
    assert len(data) == 2
    assert data[0] == ("What is 1+1?", "2", "2")
    assert data[1] == ("Is the sun hot?", "Yes", "Yes")

def test_load_gsm8k_jsonl(mock_optimizer, tmp_path):
    # Create a dummy GSM8K JSONL file
    gsm8k_file = tmp_path / "gsm8k_test.jsonl"
    content = [
        {"question": "What is 1+1?", "answer": "The answer is 2 #### 2"},
        {"question": "What is 2+2?", "answer": "The answer is 4 #### 4"}
    ]
    with open(gsm8k_file, "w", encoding="utf-8") as f:
        for item in content:
            f.write(json.dumps(item) + "\n")
    
    config = OptimizeConfig(
        dataset_path=str(gsm8k_file),
        num_examples=10
    )
    
    data = mock_optimizer._load_training_data_raw(config)
    
    assert len(data) == 2
    assert data[0] == ("What is 1+1?", "The answer is 2 #### 2", "2")
    assert data[1] == ("What is 2+2?", "The answer is 4 #### 4", "4")
