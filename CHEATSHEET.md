# Cheat sheet for our config

Documentation: <https://huggingface.co/docs/lerobot/so101>

This assumes the follower and leader symlinks are configured via udev rules:

```text
**File:** /etc/udev/rules.d/99-feetech-symlink.rule
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="YOUR SERIAL HERE", SYMLINK+="motor-bus-follower"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="YOUR SERIAL HERE", SYMLINK+="motor-bus-leader"
```

Then reload:

```bash
sudo udevadm control --reload
sudo udevadm trigger
```

## Calibrate follower

```bash
python -m lerobot.calibrate --robot.type=so101_follower --robot.port=/dev/motor-bus-follower --robot.id=follower12vblack
```

## Calibrate leader

```bash
python -m lerobot.calibrate --teleop.type=so101_leader --teleop.port=/dev/motor-bus-follower --teleop.id=leader7vblack
```

## Test teleoperation

```bash
python -m lerobot.teleoperate --teleop.type=so101_leader --teleop.port=/dev/motor-bus-leader --teleop.id=leader7vblack --robot.type=so101_follower --robot.port=/dev/motor-bus-follower --robot.id=follower12vblack
```

## Record dataset

```bash
python -m lerobot.record --robot.type=so101_follower --robot.port=/dev/motor-bus-follower --robot.cameras="{webcam: {type: opencv, index_or_path: 2, fps: 30, width: 640, height: 480}, main: {type: opencv, index_or_path: 0, fps: 30, width: 640, height: 480}}" --robot.id=follower12vblack --teleop.type=so101_leader --teleop.port=/dev/motor-bus-leader --teleop.id=leader7vblack --dataset.repo_id=aol-work/record-test --dataset.num_episodes=2 --dataset.single_task="Move the green cylinder towards the grey spot" --dataset.private=true --dataset.push_to_hub=false
```

## Continue a recording

```bash
python -m lerobot.record --robot.type=so101_follower --robot.port=/dev/motor-bus-follower --robot.cameras="{webcam: {type: opencv, index_or_path: 2, fps: 30, width: 640, height: 480}, main: {type: opencv, index_or_path: 0, fps: 30, width: 640, height: 480}}" --robot.id=follower12vblack --teleop.type=so101_leader --teleop.port=/dev/motor-bus-leader --teleop.id=leader7vblack --dataset.repo_id=aol-work/greyspot-new --dataset.num_episodes=8 --dataset.episode_time_s=12 --dataset.reset_time_s=5 --dataset.single_task="Move the green cylinder towards the grey spot" --dataset.private=true --dataset.push_to_hub=false --resume=true
```

## Inference

```bash
./eval_gr00t_on_so_101.py
```
