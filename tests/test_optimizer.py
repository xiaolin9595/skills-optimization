
import pytest
from unittest.mock import MagicMock, patch
import torch
from skill_opt.core.config import AppConfig, OptimizeConfig
from skill_opt.core.interfaces import Skill
from skill_opt.optimizer.greater import GreaterOptimizer

@pytest.fixture
def mock_components():
    with patch("skill_opt.optimizer.greater.load_model_and_tokenizer") as mock_load, \
         patch("skill_opt.optimizer.greater.greater_core") as mock_core, \
         patch("skill_opt.optimizer.greater.SkillPrompter") as mock_prompter_cls:
        
        # Mock Model & Tokenizer
        model = MagicMock()
        model.device = torch.device('cpu')
        # Mock forward pass for manual validation loop
        model.return_value.logits = torch.randn(1, 10, 200) # [Batch, Seq, Vocab]
        
        tokenizer = MagicMock()
        tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        tokenizer.decode.return_value = "Optimized Content"
        tokenizer.pad_token_id = 0
        tokenizer.vocab_size = 100
        tokenizer.all_special_ids = [0]
        
        mock_load.return_value = (model, tokenizer)
        
        # Mock Prompter Instance
        mock_prompter = MagicMock()
        mock_prompter.input_ids = torch.tensor([101, 1, 2, 3, 102]) # Mock Input
        mock_prompter._control_slice = slice(1, 4) # Covers tokens 1, 2, 3 (len 3)
        mock_prompter._loss_slice = slice(4, 5)
        mock_prompter.control = "optimized_control"
        mock_prompter.final_target = "target"
        
        mock_prompter_cls.return_value = mock_prompter
        
        # Mock Core Functions
        mock_core.propose_candidates.return_value = torch.tensor([10, 11])
        mock_core.compute_gradient.return_value = torch.tensor([[0.1, 0.2]])
        
        yield mock_load, mock_core, mock_prompter_cls

def test_optimizer_flow(mock_components):
    mock_load, mock_core, mock_prompter_cls = mock_components
    
    config = AppConfig()
    opt_config = OptimizeConfig(iterations=1, top_k=10, top_mu=2, batch_size=2)
    
    optimizer = GreaterOptimizer(config)
    
    skill = Skill(
        name="test_skill",
        description="test",
        content="Original Content"
    )
    
    # Run Optimize
    result = optimizer.optimize(skill, opt_config)
    
    # Assertions
    # Optimization loop runs for control_len = 3 tokens.
    assert result.content == "optimized_control"
    # Result optimization_metrics might depend on the mock model logits
    
    # Verification of calls
    # propose_candidates called (1 epoch * 3 positions)
    assert mock_core.propose_candidates.call_count == 3
    
    # compute_gradient called 
    # Batch size = 2 in config. data loader returns 2 samples.
    # So for each position (3), we loop batch (2). Total = 6 calls.
    assert mock_core.compute_gradient.call_count == 6
