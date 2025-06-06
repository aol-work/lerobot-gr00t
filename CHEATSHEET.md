# Cheat sheet for our config

## Calibrate follower

```bash
python -m lerobot.calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower12vblack
```

## Calibrate leader

```bash
python -m lerobot.calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader7vblack
```

## Test teleoperation

```bash
python -m lerobot.teleoperate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader7vblack --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower12vblack
```
