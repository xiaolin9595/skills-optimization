import unittest
import json
import tempfile
import os
import torch
from unittest.mock import MagicMock, patch
from skill_opt.core.config import AppConfig
from skill_opt.core.interfaces import Skill
from skill_opt.optimizer.greater import GreaterOptimizer
from skill_opt.optimizer.prompter import SkillPrompter

class TestExperimentValidation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')

    def test_llama3_slicing(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.vocab_size = 100
        
        def mock_encode(text, add_special_tokens=False):
             ids = list(range(len(text)))
             return MagicMock(input_ids=ids)

        tokenizer.side_effect = mock_encode
        
        prompter = SkillPrompter(
             goal="GOAL",
             target="TARGET",
             control_init="CONTROL",
             tokenizer=tokenizer,
             template_name="llama-3",
             device=self.device
        )
        
        self.assertEqual(prompter._user_role_slice.start, 0)
        self.assertEqual(prompter._goal_slice.start, prompter._user_role_slice.stop)
        self.assertEqual(prompter._control_slice.start, prompter._goal_slice.stop)
        self.assertEqual(prompter._assistant_role_slice.start, prompter._control_slice.stop)
        self.assertEqual(prompter._target_slice.start, prompter._assistant_role_slice.stop)
        self.assertEqual(prompter._loss_slice.start, prompter._target_slice.start - 1)

    @patch("skill_opt.optimizer.greater.load_model_and_tokenizer")
    def test_data_loading(self, mock_load):
         mock_model = MagicMock()
         mock_model.device = 'cpu'
         mock_tokenizer = MagicMock()
         mock_load.return_value = (mock_model, mock_tokenizer)
         
         items = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)]
         with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
              for item in items:
                   tmp.write(json.dumps(item) + "\n")
              tmp_path = tmp.name
              
         try:
            config = AppConfig(model_name="test")
            optimizer = GreaterOptimizer(config)
            
            class MockOptConfig:
                 dataset_path = tmp_path
                 num_examples = 3
            
            data = optimizer._load_training_data_raw(MockOptConfig())
            self.assertEqual(len(data), 3)
            self.assertEqual(data[0], ("Q0", "A0"))
            
         finally:
            os.remove(tmp_path)

    @patch("skill_opt.optimizer.greater.load_model_and_tokenizer")
    def test_evaluation_flow(self, mock_load):
         mock_model = MagicMock()
         mock_model.device = 'cpu'
         
         def generate_side_effect(input_ids, **kwargs):
              new_toks = torch.tensor([[42]], device='cpu')
              return torch.cat([input_ids, new_toks], dim=1)
         mock_model.generate.side_effect = generate_side_effect
         
         mock_tokenizer = MagicMock()
         mock_tokenizer.pad_token_id = 0
         mock_tokenizer.eos_token_id = 1
         
         def mock_decode(ids, skip_special_tokens=True):
              if 42 in ids.tolist():
                   return "#### 100"
              return ""
         mock_tokenizer.decode.side_effect = mock_decode
         mock_tokenizer.side_effect = lambda x, **k: MagicMock(input_ids=torch.tensor([1,2,3])) 
         
         mock_load.return_value = (mock_model, mock_tokenizer)
         
         item = {"question": "Q", "answer": "#### 100"}
         with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
              tmp.write(json.dumps(item) + "\n")
              tmp_path = tmp.name
              
         try:
              config = AppConfig(model_name="test")
              optimizer = GreaterOptimizer(config)
              skill = Skill(name="s", description="d", content="c")
              results = optimizer.evaluate_skill(skill, tmp_path, num_examples=1, template_name="llama-3")
              
              self.assertEqual(results['accuracy'], 1.0)
              
         finally:
              os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()
