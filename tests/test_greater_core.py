import pytest
import torch
from unittest.mock import MagicMock
from skill_opt.optimizer.greater_core import propose_candidates, generate_reasoning, compute_gradient, select_and_update

# Mock Model Output
class MockModelOutput:
    def __init__(self, logits):
        self.logits = logits

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.device = torch.device('cpu')
    
    # Mock Forward
    def forward(input_ids=None, inputs_embeds=None, **kwargs):
        batch = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        seq = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        voc = 100
        
        # Helper to maintain grad connection
        out = torch.randn(batch, seq, voc, requires_grad=True)
        if inputs_embeds is not None:
            # Fake dependency: Add sum of embeddings to output to link graph
            # inputs_embeds: [Batch, Seq, Dim]
            # We add a tiny fraction of sum to ensure gradients flow back
            out = out + inputs_embeds.sum() * 0.0001
            
        return MockModelOutput(out)
    
    model.side_effect = forward
    
    # Mock Generate
    def generate(input_ids, **kwargs):
        # Return input + 5 new tokens
        new_tokens = torch.tensor([[50, 51, 52, 53, 54]])
        return torch.cat([input_ids, new_tokens], dim=1)
    
    model.generate.side_effect = generate
    
    # Mock Embeddings
    embeds = torch.randn(100, 32) # Vocab 100, Dim 32
    embedding_layer = MagicMock()
    embedding_layer.weight = embeds
    embedding_layer.side_effect = lambda x: embeds[x]
    
    if hasattr(model, "get_input_embeddings"):
        model.get_input_embeddings.return_value = embedding_layer
    else:
        # Simulate Llama structure
        model.model = MagicMock()
        model.model.embed_tokens = embedding_layer
        
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "text"
    return tokenizer

def test_propose_candidates(mock_model, mock_tokenizer):
    prompt_ids = torch.tensor([1, 2, 3])
    input_ids_batch = torch.tensor([[4, 5], [6, 7]])
    
    # Run
    # With random logits, intersection is likely empty unless top_k is large or vocab small.
    # Our mock vocab is 100. Top K 50 -> 50% overlap.
    candidates = propose_candidates(
        mock_model, mock_tokenizer, prompt_ids, input_ids_batch, position_idx=1, top_k=80
    )
    
    assert isinstance(candidates, torch.Tensor)
    # Check shape
    assert candidates.dim() == 1

def test_generate_reasoning(mock_model, mock_tokenizer):
    prompt_ids = torch.tensor([1, 2, 3])
    input_ids = torch.tensor([4, 5])
    
    seq = generate_reasoning(mock_model, mock_tokenizer, prompt_ids, input_ids, max_new_tokens=5)
    
    # Prompt(3) + Input(2) + Generated(5) = 10
    assert seq.shape[0] == 10
    assert torch.equal(seq[:3], prompt_ids)

def test_compute_gradient(mock_model):
    # Context: 10 tokens
    context = torch.arange(10)
    candidates = torch.tensor([5, 6, 7])
    
    # Loss slice: Last 2 tokens
    loss_slice = slice(8, 10)
    
    # Mock the embedding access precisely
    mock_model.get_input_embeddings.return_value.weight.requires_grad_(True)
    
    grad = compute_gradient(
        mock_model, context, prompt_pos_idx=1, loss_slice=loss_slice, candidates=candidates
    )
    
    assert grad.shape == (1, 3) # 1 x Candidates

def test_select_and_update(mock_model, mock_tokenizer):
    prompt_ids = torch.tensor([1, 2, 3])
    candidates = torch.tensor([10, 11, 12])
    gradients = torch.tensor([[0.1, -0.5, 0.2]]) # Candidate index 1 is best (-0.5)
    
    val_in = torch.tensor([4])
    val_lbl = torch.tensor([5])
    
    best_prompt, loss = select_and_update(
        mock_model, mock_tokenizer, prompt_ids, 
        position_idx=2, 
        candidates=candidates, 
        gradients=gradients,
        validation_input_ids=val_in,
        validation_label_ids=val_lbl,
        top_mu=2
    )
    
    assert len(best_prompt) == 3
    # We expect candidate 11 (index 1) to be tested.
    # Logic: It selects top mu best gradients.
    # Best grad is -0.5 (index 1, token 11).
    # Then forward pass determines best loss.
    # Mock loss is random, so output is random from the top_mu.
