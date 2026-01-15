from typing import Optional
from pydantic import BaseModel, Field

class OptimizeConfig(BaseModel):
    """Configuration for the optimization process."""
    learning_rate: float = Field(0.01, description="Learning rate for gradient optimization")
    iterations: int = Field(10, description="Number of optimization iterations")
    device: str = Field("cuda", description="Device to use (cuda/cpu)")
    batch_size: int = Field(1, description="Batch size for gradient computation")
    top_k: int = Field(50, description="Number of candidates for logic intersection (Stage 1)")
    top_mu: int = Field(5, description="Number of best candidates to validate (Stage 4)")
    control_weight: float = Field(0.0, description="Weight for control stability loss")
    template_name: str = Field("llama-2", description="Chat Template Name (llama-2, llama-3)")
    temperature: float = Field(0.7, description="Temperature for reasoning generation")
    
    # Sequential Increasing & Early Stopping
    start_len: int = Field(20, description="Initial control length") # Reference default
    end_len: int = Field(20, description="Max control length")
    patience: int = Field(10, description="Steps to wait before increasing length or stopping")
    early_stop_threshold: float = Field(0.05, description="Loss threshold for early stopping")
    
    # Dataset
    dataset_path: Optional[str] = Field(None, description="Path to input dataset (jsonl)")
    num_examples: int = Field(5, description="Number of examples to load from dataset")
    extract_prompt: Optional[str] = Field("Therefore, the answer is", description="Prompt to induce final answer expectation")
    
    # Gradient Stability
    grad_clip: float = Field(1.0, description="Gradient clipping threshold for numerical stability")
    use_amp: bool = Field(False, description="Whether to use automatic mixed precision training")
    grad_norm_epsilon: float = Field(1e-6, description="Epsilon for gradient normalization to avoid division by zero")

from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    """Global application configuration."""
    log_level: str = Field("INFO", description="Logging level")
    environment: str = Field("development", description="Environment (development/production)")
    model_name: Optional[str] = Field(None, description="Path or name of the model")
    
    # Sub-configs
    optimization: OptimizeConfig = Field(default_factory=OptimizeConfig)

    model_config = SettingsConfigDict(
        env_prefix="SKILL_OPT_",
        env_file=".env",
        extra="ignore"
    )
