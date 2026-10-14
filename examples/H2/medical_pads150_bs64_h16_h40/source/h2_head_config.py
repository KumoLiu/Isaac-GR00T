"""H2 + Sharpa: one head RGB image, joint state, language, no tactile inputs."""

import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


JOINT_GROUPS = ["left_arm", "right_arm", "left_hand", "right_hand"]
ACTION_HORIZON = int(os.environ.get("MEDICAL_ACTION_HORIZON", "40"))


def make_h2_head_config(action_horizon=ACTION_HORIZON):
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["high"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=JOINT_GROUPS),
        "action": ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=JOINT_GROUPS,
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE
                    if "arm" in key
                    else ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
                for key in JOINT_GROUPS
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }


h2_head_config = make_h2_head_config()
register_modality_config(h2_head_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
