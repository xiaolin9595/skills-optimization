from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .config import OptimizeConfig

# --- Data Models ---

class Skill(BaseModel):
    """Represents an agent skill."""
    name: str
    description: str
    content: str  # The actual prompt/instructions
    metadata: Dict[str, Any] = Field(default_factory=dict)

class OptimizedSkill(Skill):
    """A skill that has been optimized."""
    optimization_metrics: Dict[str, float] = Field(default_factory=dict)
    original_skill_name: str

class AdaptedSkill(Skill):
    """A skill adapted for a specific target model."""
    target_model: str
    source_optimization_id: Optional[str] = None

class Task(BaseModel):
    """A task to execute for evaluation."""
    id: str
    instruction: str
    expected_output: Optional[str] = None
    input_data: Dict[str, Any] = Field(default_factory=dict)

class ExecutionResult(BaseModel):
    """Result of executing a skill on a task."""
    task_id: str
    success: bool
    output: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

# --- Interfaces ---

class SkillOptimizer(ABC):
    """Interface for skill optimization (e.g., GreaTer)."""
    
    @abstractmethod
    def optimize(self, skill: Skill, config: OptimizeConfig) -> OptimizedSkill:
        """
        Optimizes a skill using gradient-based or other methods.
        
        Args:
            skill: The input skill to optimize.
            config: Configuration for the optimization process.
            
        Returns:
            The optimized skill.
        """
        pass

class SkillBridge(ABC):
    """Interface for cross-model skill transfer (e.g., PromptBridge)."""
    
    @abstractmethod
    def transfer(self, skill: OptimizedSkill, target_model: str) -> AdaptedSkill:
        """
        Adapts an optimized skill for a different target model.
        
        Args:
            skill: The optimized skill from the source model.
            target_model: Identifier for the target model.
            
        Returns:
            The adapted skill.
        """
        pass

class SkillExecutor(ABC):
    """Interface for executing skills (e.g., via iFlow)."""
    
    @abstractmethod
    def execute(self, skill: AdaptedSkill, task: Task) -> ExecutionResult:
        """
        Executes a task using the provided skill.
        
        Args:
            skill: The skill to use.
            task: The task to perform.
            
        Returns:
            The result of the execution.
        """
        pass
