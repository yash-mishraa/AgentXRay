"""Model-backed pilot benchmark tasks and ground-truth specs."""

from benchmarks.model_pilot.ground_truth import (
    ModelPilotGroundTruthSpec,
    evaluate_model_pilot_ground_truth,
)
from benchmarks.model_pilot.runner import (
    dry_run_model_pilot,
    run_model_pilot_benchmark,
)
from benchmarks.model_pilot.tasks import (
    create_model_pilot_baseline_task,
    create_model_pilot_conflicting_prompt_task,
    create_model_pilot_distractor_task,
    get_all_model_pilot_tasks,
    get_model_pilot_tasks,
)

__all__ = [
    "ModelPilotGroundTruthSpec",
    "evaluate_model_pilot_ground_truth",
    "create_model_pilot_baseline_task",
    "create_model_pilot_conflicting_prompt_task",
    "create_model_pilot_distractor_task",
    "get_all_model_pilot_tasks",
    "get_model_pilot_tasks",
    "dry_run_model_pilot",
    "run_model_pilot_benchmark",
]
