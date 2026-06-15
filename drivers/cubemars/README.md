# CubeMars AK-Series Driver

CAN-bus driver for the **AK80-8-KV60** (shoulder) and **AK60-6-KV80** (elbow)
motors mounted on the cup-manipulator project.

See the full implementation note:
[`notes_all/notes_cup_manipulator_tendon/hardware/CubeMars_Hardware_Implementation.tex`](../../notes_all/notes_cup_manipulator_tendon/hardware/CubeMars_Hardware_Implementation.tex)

## Hardware

| Component | Part |
|-----------|------|
| Shoulder  | CubeMars `CM-06-03-AK80-8-KV60-With-Driver` (CAN id `0x01`) |
| Elbow     | CubeMars `CM-05-02-AK60-6-KV80-V3.0-D` (CAN id `0x02`) |
| Compute   | NVIDIA Jetson Orin Nano (J17 CAN, 3.3 V logic) |
| Transceiver | SN65HVD230 (3.3 V) |
| Power     | 24 V supply, separate from Jetson PSU |

## Bus bring-up (Jetson)

```bash
sudo modprobe mttcan
sudo ip link set can0 type can bitrate 1000000 berr-reporting on
sudo ip link set can0 up
ip -details link show can0
```

## Quick API

```python
import time
import numpy as np
from drivers.cubemars import TwoJointArm

with TwoJointArm("can0") as arm:
    while True:
        q  = arm.q()
        qd = arm.qd()
        # ... compute tau_ff with the existing computed-torque controller ...
        arm.command(q_des=q, qd_des=np.zeros(2), tau_ff=np.zeros(2))
        if not arm.healthy():
            raise RuntimeError("CAN timeout")
        time.sleep(0.002)
```

## Layout

```
drivers/cubemars/
  __init__.py
  config.py        # MotorConfig dataclass + AK80_8 / AK60_6 catalogue
  protocol.py      # MIT (force) + servo frame encode/decode
  can_iface.py     # python-can SocketCAN wrapper
  motor.py         # single-motor state + safety
  arm.py           # 2-motor TwoJointArm + rx thread
  examples/
    01_ping.py            # bring-up: read feedback only
    02_position_step.py   # servo position-loop step (no impedance)
    03_mit_hold.py        # MIT hold at current pose with low gains
    04_track_traj.py      # MIT + computed-torque feed-forward
```

## Recommended control mode

**MIT (force) mode** for both joints. The on-board impedance acts as a safety
spring-damper while host-side computed-torque does the tracking via `tau_ff`.

| Joint | `Kp` | `Kd` | Notes |
|-------|------|------|-------|
| Shoulder (AK80-8) | 30 | 1.5 | soft hold |
| Elbow    (AK60-6) | 15 | 0.5 | weaker because cable-driven |

## Outstanding info to be supplied

- Final CAN IDs (assumed `0x01`, `0x02`).
- Sign / offset for motor zero relative to the URDF `q1`, `q2`.
- Pulley ratio if the elbow is not driven 1:1.
- 24 V supply current limit.
