"""
Prompt Loading Utilities for Research Loop
==========================================

Functions for loading prompt templates from the prompts directory.
"""

import shutil
from pathlib import Path


PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = PROMPTS_DIR / f"{name}.md"
    return prompt_path.read_text()


def get_initializer_prompt() -> str:
    """Load the research initializer prompt."""
    return load_prompt("initializer_prompt")


def get_research_prompt() -> str:
    """Load the research agent prompt."""
    return load_prompt("research_prompt")


def copy_spec_to_project(project_dir: Path) -> None:
    """Copy the research spec file into the project directory for the agent to read."""
    spec_source = PROMPTS_DIR / "research_spec.md"
    spec_dest = project_dir / "research_loop" / "prompts" / "research_spec.md"
    # Spec already in place for this project structure
    if not spec_dest.exists():
        spec_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(spec_source, spec_dest)
        print("Copied research_spec.md to project directory")
