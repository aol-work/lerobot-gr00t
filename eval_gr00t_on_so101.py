import time

import numpy as np
import torch
from tqdm import tqdm

from lerobot.common.cameras.configs import ColorMode, Cv2Rotation
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.robots.so101_follower import SO101FollowerConfig, SO101Follower


# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
from gr00t.eval.service import ExternalRobotInferenceClient


MODALITY_KEYS = ["single_arm", "gripper"]
ACTION_HORIZON = 12
ACTIONS_TO_EXECUTE = 100

class Gr00tRobotInferenceClient:
    """
    This class is used to get the action from the Gr00t robot.
    """

    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction="Move the green cylinder towards the grey spot",
    ):
        self.language_instruction = language_instruction
        # 480, 640
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, img, state):
        """
        Get the action from the Gr00t robot.
        """
        obs_dict = {
            "video.webcam": img[np.newaxis, :, :, :],
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": [self.language_instruction],
        }
        res = self.policy.get_action(obs_dict)
        # print("Inference query time taken", time.time() - start_time)
        return res

    def sample_action(self):
        """
        Sample an action from the Gr00t robot.
        """
        obs_dict = {
            "video.webcam": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "state.single_arm": np.zeros((1, 5)),
            "state.gripper": np.zeros((1, 1)),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)

    def set_lang_instruction(self, lang_instruction):
        """
        Set the language instruction for the Gr00t robot.
        """
        self.language_instruction = lang_instruction


MOTOR_POSITION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def numpy_to_dict(tensor: np.ndarray) -> dict:
    """
    Convert a tensor to a dictionary.
    """
    return {MOTOR_POSITION_NAMES[i]: tensor[i] for i in range(len(MOTOR_POSITION_NAMES))}


def dict_to_numpy(d: dict) -> np.ndarray:
    """
    Convert a dictionary to a tensor.
    """
    return np.array([d[key] for key in MOTOR_POSITION_NAMES])


def move_to_initial_pose(f: SO101Follower):
    """
    Move the robot to the initial pose.
    """
    current_motor_state = numpy_to_dict(np.array([0, -90, 90, 90, 0, 0]))
    f.send_action(current_motor_state)
    time.sleep(1)
    print("-------------------------------- moving to initial pose")


config = SO101FollowerConfig(
    port="/dev/motor-bus-follower",
    id="follower12vblack",
    cameras={
        "webcam": OpenCVCameraConfig(
            index_or_path=2,
            fps=30,
            width=640,
            height=480,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
        )
    },
)

# TODO(aol-work): Get optional config from the command line
client = Gr00tRobotInferenceClient()
follower = SO101Follower(config)
follower.connect(calibrate=False)

move_to_initial_pose(follower)

try:
    for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
        observation = follower.get_observation()
        img = observation["webcam"]
        state = dict_to_numpy(observation)
        action = client.get_action(img, state)
        start_time = time.time()
        for i in range(ACTION_HORIZON):
            concat_action = np.concatenate(
                [np.atleast_1d(action[f"action.{key}"][i]) for key in MODALITY_KEYS],
                axis=0,
            )
            assert concat_action.shape == (6,), concat_action.shape
            follower.send_action(numpy_to_dict(concat_action))
            time.sleep(0.02)

            # 0.05*16 = 0.8 seconds
            print("executing action", i, "time taken", time.time() - start_time)
        print("Action chunk execution time taken", time.time() - start_time)
except KeyboardInterrupt:
    print("\nGracefully exiting on CTRL-C")

move_to_initial_pose(follower)

follower.disconnect()
