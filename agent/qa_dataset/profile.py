"""Q&A dataset profile: load, validate, resolve, and initialize.

The profile (qa_dataset_profile.yaml) is the per-dataset policy: which catalog
strategies are enabled, the question-type/difficulty mixes, verification and
dedup thresholds, coverage caps, generation settings, and output format. The
shipped default_profile.yaml carries best-practice defaults; a project copy is
the source of truth once created and is model- and user-editable.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

_PKG_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_FILENAME = "qa_dataset_profile.yaml"
DEFAULT_PERSONAS_FILENAME = "qa_personas.yaml"


def shipped_default_profile_path() -> Path:
    return _PKG_DIR / "default_profile.yaml"


def shipped_personas_path() -> Path:
    return _PKG_DIR / "default_personas.yaml"


class _Base(BaseModel):
    # Forgiving: tolerate unknown keys so hand-edited profiles never hard-fail.
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class StrategyRef(_Base):
    id: str
    weight: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class StrategySelection(_Base):
    enabled: list[StrategyRef] = Field(default_factory=list)


class SourcesCfg(_Base):
    input_dir: str = "./books"
    chunk_tokens: int = 800
    chunk_overlap: int = 100
    topic_tags: list[str] = Field(default_factory=list)
    preserve_tables: bool = True
    keep_figure_captions: bool = True
    min_chunk_chars: int = 200


class HoldoutCfg(_Base):
    strategy: str = "by_source"
    fraction: float = 0.15


class PersonasCfg(_Base):
    file: str = DEFAULT_PERSONAS_FILENAME
    version: int = 1
    sample_per_item: int = 1


class CoverageCfg(_Base):
    max_items_per_chunk: int = 8
    min_chunk_coverage: float = 0.7


class GenerationCfg(_Base):
    # Decoupled from the interactive agent: any litellm-compatible id, including
    # local prefixes (vllm/, ollama/, lm_studio/, llamacpp/). The special value
    # "session" means the interactive agent's model; "generator" used elsewhere
    # refers back to this id.
    generator_model: str = "vllm/Qwen2.5-14B-Instruct-AWQ"
    constrained_decoding: str = "outlines"  # outlines | xgrammar | none
    max_new_tokens: int = 512


class OpenEndedRubricCfg(_Base):
    enabled: bool = True
    judge_model: str = "generator"  # "generator" reuses generator_model; or an explicit id
    dimensions: list[str] = Field(
        default_factory=lambda: ["completeness", "relevance", "no_hallucinated_detail"]
    )
    min_score: int = 4


class VerificationCfg(_Base):
    schema_check: str | bool = Field(default="required", alias="schema")
    verbatim_span_check: bool = True
    faithfulness: str | bool = "required"
    answer_relevance: str | bool = "required"
    self_consistency_k: int = 5
    open_ended_rubric: OpenEndedRubricCfg = Field(default_factory=OpenEndedRubricCfg)
    max_repair_retries: int = 2


class DifficultyCalibrationCfg(_Base):
    enabled: bool = True
    base_model: str = "target"
    sample_size: int = 200


class DedupCfg(_Base):
    minhash_jaccard: float = 0.7
    embedding_cosine: float = 0.92


class DecontamCfg(_Base):
    ngram: int = 13
    embedding_cosine: float = 0.85


class DiversityGatesCfg(_Base):
    max_single_type_share: float = 0.35
    min_distinct_2: float = 0.6


class OutputCfg(_Base):
    format: str = "sft_chat"
    split_by: str = "source"
    target_size: int | str = "auto"
    system_prompt: str = (
        "You answer questions using only the provided source material. "
        "If the material does not contain the answer, say so."
    )
    abstention_text: str = "That isn't covered in the source material."
    completion_only_loss: bool = True


class QADatasetProfile(_Base):
    seed: int = 42
    catalog_version: int = 2
    strategies: StrategySelection = Field(default_factory=StrategySelection)
    sources: SourcesCfg = Field(default_factory=SourcesCfg)
    holdout: HoldoutCfg = Field(default_factory=HoldoutCfg)
    personas: PersonasCfg = Field(default_factory=PersonasCfg)
    question_types: dict[str, float] = Field(default_factory=dict)
    difficulty_mix: dict[str, float] = Field(default_factory=dict)
    coverage: CoverageCfg = Field(default_factory=CoverageCfg)
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    verification: VerificationCfg = Field(default_factory=VerificationCfg)
    difficulty_calibration: DifficultyCalibrationCfg = Field(
        default_factory=DifficultyCalibrationCfg
    )
    dedup: DedupCfg = Field(default_factory=DedupCfg)
    decontamination: DecontamCfg = Field(default_factory=DecontamCfg)
    diversity_gates: DiversityGatesCfg = Field(default_factory=DiversityGatesCfg)
    output: OutputCfg = Field(default_factory=OutputCfg)

    def enabled_strategy_ids(self) -> list[str]:
        return [s.id for s in self.strategies.enabled]

    def warnings(self) -> list[str]:
        """Non-fatal sanity checks surfaced to the user/agent."""
        out: list[str] = []
        if self.question_types:
            total = sum(self.question_types.values())
            if abs(total - 1.0) > 0.02:
                out.append(f"question_types sum to {total:.2f}, expected ~1.0")
        if self.difficulty_mix:
            total = sum(self.difficulty_mix.values())
            if abs(total - 1.0) > 0.02:
                out.append(f"difficulty_mix sums to {total:.2f}, expected ~1.0")
        if self.question_types.get("unanswerable", 0) < 0.05:
            out.append("unanswerable share < 5% — model may not learn to abstain")
        return out


def load_profile(path: str | Path) -> QADatasetProfile:
    """Load and validate a profile YAML."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return QADatasetProfile.model_validate(data)


def resolve_profile(working_dir: str | Path = ".") -> tuple[QADatasetProfile, Path]:
    """Return (profile, path). Working-dir copy wins; else the shipped default."""
    local = Path(working_dir) / DEFAULT_PROFILE_FILENAME
    path = local if local.exists() else shipped_default_profile_path()
    return load_profile(path), path


def init_project(working_dir: str | Path = ".", *, overwrite: bool = False) -> dict[str, str]:
    """Seed a project with profile + personas copied from the shipped defaults.

    Returns a map of {what: path}. Existing files are preserved unless overwrite.
    """
    wd = Path(working_dir)
    wd.mkdir(parents=True, exist_ok=True)
    created: dict[str, str] = {}

    prof_dst = wd / DEFAULT_PROFILE_FILENAME
    if overwrite or not prof_dst.exists():
        shutil.copyfile(shipped_default_profile_path(), prof_dst)
        created["profile"] = str(prof_dst)

    pers_dst = wd / DEFAULT_PERSONAS_FILENAME
    if overwrite or not pers_dst.exists():
        shutil.copyfile(shipped_personas_path(), pers_dst)
        created["personas"] = str(pers_dst)

    return created


def patch_profile(path: str | Path, updates: dict[str, Any]) -> QADatasetProfile:
    """Shallow-merge top-level updates into a profile file and rewrite it.

    Used by the qa_dataset tool so the model can adjust parameters during a
    discussion. Re-validates before writing; raises on invalid result.
    """
    p = Path(path)
    data = yaml.safe_load(p.read_text()) or {}
    data.update(updates)
    prof = QADatasetProfile.model_validate(data)  # validate before writing
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return prof
