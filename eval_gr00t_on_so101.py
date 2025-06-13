"""
This script is an example to run the Gr00t robot on the SO101 follower.
"""

import argparse
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
MOTOR_POSITION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


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


def numpy_to_dict(tensor: np.ndarray) -> dict:
    """
    Convert a tensor containing motor positions to a dictionary.
    """
    return {MOTOR_POSITION_NAMES[i]: tensor[i] for i in range(len(MOTOR_POSITION_NAMES))}


def dict_to_numpy(d: dict) -> np.ndarray:
    """
    Convert a dictionary containing motor positions to a tensor.
    """
    return np.array([d[key] for key in MOTOR_POSITION_NAMES])


def go_home(follower: SO101Follower):
    """
    Move the robot to the initial pose.
    """
    current_motor_state = numpy_to_dict(np.array([0, -90, 90, 90, 0, 0]))
    follower.send_action(current_motor_state)
    time.sleep(1)
    print("-------------------------------- moving to initial pose")


def main():
    """
    Main function to run the Gr00t robot on the SO101 follower.
    """
    parser = argparse.ArgumentParser(
        description="Run the Gr00t robot on the SO101 follower")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Host for the Gr00t robot service")
    parser.add_argument("--port", type=int, default=5555,
                        help="Port for the Gr00t robot service")
    parser.add_argument("--action_horizon", type=int, default=12,
                        help="Number of actions to execute per step")
    parser.add_argument("--actions_to_execute", type=int,
                        default=100, help="Total number of actions to execute")
    parser.add_argument("--camera_idx", type=int,
                        default=2, help="Camera index to use")
    parser.add_argument("--lang_instruction", type=str, default="Move the green cylinder towards the grey spot",
                        help="Language instruction for the robot")
    args = parser.parse_args()

    # Print language instruction for confirmation
    print("Language instruction:", args.lang_instruction)

    config = SO101FollowerConfig(
        port="/dev/motor-bus-follower",
        id="follower12vblack",
        cameras={
            "webcam": OpenCVCameraConfig(
                index_or_path=args.camera_idx,
                fps=30,
                width=640,
                height=480,
                color_mode=ColorMode.RGB,
                rotation=Cv2Rotation.NO_ROTATION,
            )
        },
    )

    client = Gr00tRobotInferenceClient(
        host=args.host,
        port=args.port,
        language_instruction=args.lang_instruction
    )
    follower = SO101Follower(config)
    # TODO(aoldemeier): Add optional calibration
    follower.connect(calibrate=False)
    # TODO(aoldemeier): Load custom preset from original example

    # TODO(aoldemeier): Move to initial position instead
    go_home(follower)

    try:
        for i in tqdm(range(args.actions_to_execute), desc="Executing actions"):
            observation = follower.get_observation()
            img = observation["webcam"]
            state = dict_to_numpy(observation)
            action = client.get_action(img, state)
            start_time = time.time()
            for i in range(args.action_horizon):
                concat_action = np.concatenate(
                    [np.atleast_1d(action[f"action.{key}"][i])
                     for key in MODALITY_KEYS],
                    axis=0,
                )
                assert concat_action.shape == (6,), concat_action.shape
                follower.send_action(numpy_to_dict(concat_action))
                time.sleep(0.02)

                # 0.05*16 = 0.8 seconds
                print("executing action", i, "time taken",
                      time.time() - start_time)
            print("Action chunk execution time taken", time.time() - start_time)
    except KeyboardInterrupt:
        print("\nGracefully exiting on CTRL-C")

    go_home(follower)

    follower.disconnect()


if __name__ == "__main__":
    main()
