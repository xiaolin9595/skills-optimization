import pytest
from skill_opt.core.config import AppConfig, OptimizeConfig
from skill_opt.core.logger import setup_logger
from skill_opt.core.interfaces import (
    Skill, OptimizedSkill, AdaptedSkill, 
    SkillOptimizer, SkillBridge, SkillExecutor
)

def test_config_defaults():
    config = AppConfig()
    assert config.log_level == "INFO"
    assert config.optimization.iterations == 10

def test_logger_setup():
    logger = setup_logger("test_logger", "DEBUG")
    assert logger.level == 10  # DEBUG level

def test_skill_models():
    skill = Skill(name="test", description="desc", content="Do X")
    assert skill.name == "test"
    
    opt_skill = OptimizedSkill(
        name="test", description="desc", content="Do X better",
        original_skill_name="test",
        optimization_metrics={"loss": 0.5}
    )
    assert opt_skill.optimization_metrics["loss"] == 0.5

def test_interfaces_are_abstract():
    # Trying to instantiate ABC should verify we can't without implementation
    with pytest.raises(TypeError):
        SkillOptimizer()

class MockOptimizer(SkillOptimizer):
    def optimize(self, skill, config):
        return OptimizedSkill(
            name=skill.name, description=skill.description, 
            content=skill.content, original_skill_name=skill.name
        )

def test_mock_implementation():
    opt = MockOptimizer()
    skill = Skill(name="s", description="d", content="c")
    res = opt.optimize(skill, OptimizeConfig())
    assert isinstance(res, OptimizedSkill)
