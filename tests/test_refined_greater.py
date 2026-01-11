import pytest
import torch
import random
from unittest.mock import MagicMock
from skill_opt.optimizer.greater_core import propose_candidates, compute_gradient
from skill_opt.optimizer.prompter import SkillPrompter

def test_propose_candidates_union_intersection():
    # Setup Mock Model
    model = MagicMock()
    model.device = torch.device('cpu')
    
    # Mock Logits: [Batch=4, Seq=5, Vocab=10]
    # We want to force specific Top-K scenarios.
    # Pos = 3. Logits at 2 predict 3.
    # Sample 0 TopK: {1, 2, 3}
    # Sample 1 TopK: {2, 3, 4}
    # Sample 2 TopK: {5, 6, 7}
    # Sample 3 TopK: {6, 7, 8}
    
    # Intersection(0, 1) = {2, 3}
    # Intersection(2, 3) = {6, 7}
    # Union = {2, 3, 6, 7}
    
    # We fake logits by creating a tensor where desired indices have high values
    batch_size = 4
    vocab_size = 10
    seq_len = 5
    top_k = 3
    
    logits = torch.zeros(batch_size, seq_len, vocab_size)
    
    # Set high values for target position (idx 2)
    pos_idx = 3
    logit_idx = pos_idx - 1 # 2
    
    # Sample 0: 1, 2, 3
    logits[0, logit_idx, [1, 2, 3]] = 10.0
    # Sample 1: 2, 3, 4
    logits[1, logit_idx, [2, 3, 4]] = 10.0
    # Sample 2: 5, 6, 7
    logits[2, logit_idx, [5, 6, 7]] = 10.0
    # Sample 3: 6, 7, 8
    logits[3, logit_idx, [6, 7, 8]] = 10.0
    
    model.return_value.logits = logits
    
    tokenizer = MagicMock()
    prompt = torch.zeros(5, dtype=torch.long)
    input_ids_batch = torch.zeros(batch_size, 5, dtype=torch.long)
    
    # Run
    candidates = propose_candidates(
        model, tokenizer, prompt, input_ids_batch, 
        position_idx=pos_idx, 
        top_k=top_k, 
        num_intersections=10, # Enough to hit the pairs
        intersection_across_examples=2
    )
    
    # We expect {2, 3, 6, 7} to be in the set. 
    # Logic: It tries random pairs.
    # Pair (0,1) -> {2,3}. Pair (2,3) -> {6,7}.
    # Pair (0,2) -> {} likely.
    # Union should contain {2, 3, 6, 7}.
    
    cand_list = candidates.tolist()
    assert 2 in cand_list
    assert 3 in cand_list
    assert 6 in cand_list
    assert 7 in cand_list
    # 4 and 5 might be missing if (1,2) never picked or empty intersection.
    
    
def test_skill_prompter_llama2():
    tokenizer = MagicMock()
    # Mock tokenizer to return length = len(text.split())
    # And input_ids = [1] * len
    tokenizer.side_effect = lambda x, **kwargs: MagicMock(input_ids=[1]*len(x.split()))
    
    prompter = SkillPrompter(
        goal="Goal",
        target="Target",
        tokenizer=tokenizer,
        template_name='llama-2',
        control_init="Control"
    )
    
    # Structure: [INST] Goal Control [/INST] Target
    # Len:
    # [INST] = 1, Goal = 1
    # Control = 1
    # [/INST] = 1
    # Target = 1
    # Total = 5
    
    # Check Slices
    # User: 0..0 ? (Base is empty string -> len 0)
    # Goal: starts after user stuff. 
    # Note: Our Mock is crude.
    # Let's verify slice ordering.
    
    assert prompter._goal_slice.start <= prompter._goal_slice.stop
    assert prompter._control_slice.start == prompter._goal_slice.stop
    assert prompter._target_slice.start >= prompter._assistant_role_slice.stop
    assert prompter._loss_slice.stop == prompter._target_slice.stop - 1

