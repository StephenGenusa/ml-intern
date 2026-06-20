"""ml-intern Q&A dataset generation pipeline.

Turns a corpus of documents into a grounded, diverse, verified question-answer
dataset in SFT chat format. Driven by a model-editable qa_dataset_profile.yaml
that selects strategies from strategies/catalog.yaml. See QA_PIPELINE_INTEGRATION_PLAN.md.
"""

from agent.qa_dataset.profile import (  # noqa: F401
    QADatasetProfile,
    load_profile,
    resolve_profile,
    init_project,
    patch_profile,
    shipped_default_profile_path,
    shipped_personas_path,
)
from agent.qa_dataset.strategies import (  # noqa: F401
    Strategy,
    Catalog,
    load_catalog,
    resolve_enabled_strategies,
    validate_selection,
    generators_for_type,
    shipped_catalog_path,
)

__all__ = [
    "QADatasetProfile",
    "load_profile",
    "resolve_profile",
    "init_project",
    "patch_profile",
    "shipped_default_profile_path",
    "shipped_personas_path",
    "Strategy",
    "Catalog",
    "load_catalog",
    "resolve_enabled_strategies",
    "validate_selection",
    "generators_for_type",
    "shipped_catalog_path",
]
