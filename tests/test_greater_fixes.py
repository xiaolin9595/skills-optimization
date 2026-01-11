import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from skill_opt.optimizer import greater_core
from skill_opt.optimizer.greater import GreaterOptimizer
from skill_opt.core.config import AppConfig, OptimizeConfig
from skill_opt.core.interfaces import Skill
from skill_opt.optimizer.utils import get_nonascii_toks

class TestGreaterFixes(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        
    def test_get_nonascii_toks(self):
        # Mock Tokenizer
        tokenizer = MagicMock()
        tokenizer.vocab_size = 5
        tokenizer.all_special_ids = [4]
        
        # 0: "a" (ascii)
        # 1: "好" (non-ascii)
        # 2: "b" (ascii)
        # 3: "\x00" (non-printable)
        # 4: [EOS]
        
        def mock_decode(ids):
            mapping = {0: "a", 1: "好", 2: "b", 3: "\x00", 4: ""}
            return mapping.get(ids[0], "")
            
        tokenizer.decode.side_effect = mock_decode
        
        non_ascii_tensor = get_nonascii_toks(tokenizer, device=self.device)
        non_ascii_list = sorted(non_ascii_tensor.tolist())
        
        # Expected: 1 ("好"), 3 (non-printable), 4 (special)
        self.assertEqual(non_ascii_list, [1, 3, 4])

    def test_focused_loss_logic(self):
        # Test compute_gradient with focused_target.
        # We need a model where outputs depend on inputs for backward() to work.
        
        # Simple Mock Model acting like a linear layer
        linear = nn.Linear(4, 10) # Dim 4 -> Vocab 10
        model = MagicMock()
        model.device = self.device
        model.get_input_embeddings.return_value = nn.Embedding(10, 4)
        
        def forward(inputs_embeds=None, **kwargs):
            # inputs_embeds: [1, Seq, 4]
            # logits: [1, Seq, 10]
            if inputs_embeds is None:
                 # Fallback for initial check
                 return MagicMock(logits=torch.randn(1, 3, 10))
            return MagicMock(logits=linear(inputs_embeds))
            
        model.side_effect = forward
        
        candidates = torch.tensor([1], device=self.device)
        context_ids = torch.tensor([1, 2, 3], device=self.device)
        loss_slice = slice(1, 3) 
        focused_target = torch.tensor([5, 6], device=self.device)
        
        # We interpret logits and ensure focused logic doesn't crash
        # and returns a gradient tensor.
        
        grad = greater_core.compute_gradient(
            model=model,
            context_ids_with_gt=context_ids,
            prompt_pos_idx=0,
            loss_slice=loss_slice,
            candidates=candidates,
            focused_target=focused_target,
            device=self.device
        )
        
        self.assertTrue(torch.is_tensor(grad))
        self.assertEqual(grad.shape, (1, 1))

    @patch("skill_opt.optimizer.greater.SkillPrompter")
    @patch("skill_opt.optimizer.greater.load_model_and_tokenizer")
    def test_optimizer_sequential_increasing(self, mock_load, mock_prompter_cls):
        # Setup mocks
        model = MagicMock()
        model.device = self.device
        model.return_value.logits = torch.randn(1, 10, 10) 
        
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.decode.return_value = "control"
        tokenizer.vocab_size = 10
        tokenizer.all_special_ids = []
        
        mock_load.return_value = (model, tokenizer)
        
        # Prompter instance
        mock_prompter = MagicMock()
        mock_prompter.input_ids = torch.tensor([1, 2, 3], device=self.device)
        mock_prompter._control_slice = slice(0, 1) # Len 1
        mock_prompter._loss_slice = slice(1, 2)
        mock_prompter.control = "init"
        
        mock_prompter_cls.return_value = mock_prompter
        
        config = AppConfig()
        opt_config = OptimizeConfig(
            iterations=1, 
            start_len=1, 
            end_len=2, 
            patience=1, 
            early_stop_threshold=-1.0, 
            top_k=2,
            top_mu=1, # Fix: must be <= top_k (2) and available candidates
            batch_size=1
        )
        
        optimizer = GreaterOptimizer(config)
        skill = Skill(name="test", description="desc", content="init")
        
        # Mock propose candidates to return something
        with patch("skill_opt.optimizer.greater.greater_core.propose_candidates") as mock_prop, \
             patch("skill_opt.optimizer.greater.greater_core.compute_gradient") as mock_grad:
             
             mock_prop.return_value = torch.tensor([1, 2], device=self.device)
             mock_grad.return_value = torch.tensor([[0.1, 0.2]], device=self.device)
             
             result = optimizer.optimize(skill, opt_config)
             
        # Assertions
        # Should have iterated length 1 and 2.
        # SkillPrompter init calls:
        # 1. Initial creation (len 1)
        # 2. Sequential update (len 2)
        # Plus optimize loop creates prompters every length_iter.
        # Total inits > 1
        self.assertGreater(mock_prompter_cls.call_count, 1)
        
        # Check "Increasing control length" in logs not easily possible here, 
        # but the flow suggests it ran.

if __name__ == '__main__':
    unittest.main()
