"""YAM constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflect_rotary_to_linear,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

YAM_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "i2rt_yam" / "xmls" / "yam.xml"
)
assert YAM_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, YAM_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(YAM_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Effective inertias: M[dof, dof] from mj_fullM at qpos0, including reflected
# motor armature. left_finger includes both fingers via the equality constraint.
# Recompute if the model geometry or transmission changes (see __main__ below).
EFFECTIVE_INERTIAS = {
  "joint1": 0.123153,
  "joint2": 0.277411,
  "joint3": 0.232763,
  "joint4": 0.030154,
  "joint5": 0.009126,
  "joint6": 0.002868,
  "left_finger": 8.609214,
}

ARMATURE_DM_4340 = 0.032
ARMATURE_DM_4310 = 0.0018

# Reference: https://github.com/i2rt-robotics/i2rt/blob/cbe48976b44aae45af856c62545be00ea2feed11/i2rt/motor_drivers/utils.py#L159-L169
DM_4340 = ElectricActuator(
  reflected_inertia=ARMATURE_DM_4340,
  velocity_limit=10.0,
  effort_limit=28.0,
)
# Reference: https://github.com/i2rt-robotics/i2rt/blob/cbe48976b44aae45af856c62545be00ea2feed11/i2rt/motor_drivers/utils.py#L139-L149
DM_4310 = ElectricActuator(
  reflected_inertia=ARMATURE_DM_4310,
  velocity_limit=30.0,
  effort_limit=10.0,
)

NATURAL_FREQ_TARGET = 10.0 * 2.0 * 3.1415926535  # 10 Hz target
DAMPING_RATIO = 1.0

# Hardware kd limits (after torque constant correction of 1.3x).
TORQUE_CONSTANT_CORRECTION = 1.3
_KD_HW_MAX = {
  "DM4340": 5.0,
  "DM4310": 2.5,
}

# Per-joint PD gains: use target frequency, clamped per-joint so that
# kd_hw = kd_sim / TORQUE_CONSTANT_CORRECTION <= kd_hw_max.
# No kp limit (only kd wraps on 12-bit encoding).
_ARM_JOINTS: dict[str, ElectricActuator] = {
  "joint1": DM_4340,
  "joint2": DM_4340,
  "joint3": DM_4340,
  "joint4": DM_4310,
  "joint5": DM_4310,
  "joint6": DM_4310,
}
_ARM_MOTOR_TYPE: dict[str, str] = {
  "joint1": "DM4340",
  "joint2": "DM4340",
  "joint3": "DM4340",
  "joint4": "DM4310",
  "joint5": "DM4310",
  "joint6": "DM4310",
}


def _clamped_omega(name: str) -> float:
  """Natural frequency for a joint, clamped to its hardware kd limit."""
  m = EFFECTIVE_INERTIAS[name]
  kd_sim_max = _KD_HW_MAX[_ARM_MOTOR_TYPE[name]] * TORQUE_CONSTANT_CORRECTION
  omega_kd = kd_sim_max / (2.0 * DAMPING_RATIO * m)
  return min(NATURAL_FREQ_TARGET, omega_kd)


# Resulting per-joint frequencies (Hz):
#   joint1: 4.20, joint2: 1.86, joint3: 2.22,
#   joint4: 8.58, joint5: 10.00, joint6: 10.00
ARM_ACTUATORS = tuple(
  BuiltinPositionActuatorCfg(
    target_names_expr=(name,),
    stiffness=EFFECTIVE_INERTIAS[name] * _clamped_omega(name) ** 2,
    damping=2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS[name] * _clamped_omega(name),
    effort_limit=motor.effort_limit,
    armature=motor.reflected_inertia,
  )
  for name, motor in _ARM_JOINTS.items()
)

##
# Gripper transmission parameters (linear_4310).
#
# The DM4310 motor drives a linear mechanism (gear radius ~14.6 mm) that
# converts rotation directly to finger motion. The transmission ratio is
# constant, unlike the crank which varies with angle.
#
# The MuJoCo model uses a linear slide joint for left_finger (meters).
# reflect_rotary_to_linear converts the rotary motor specs (rad, Nm) to
# equivalent linear specs (m, N) using energy/power equivalence:
#   armature [kg]  = I_motor / r^2
#   vel_limit [m/s] = omega_limit * r
#   force_limit [N] = torque_limit / r
# where r = gripper_stroke / motor_stroke [m/rad].
##

# Reference: i2rt GripperType.LINEAR_4310 parameters.
GRIPPER_MOTOR_STROKE = 6.57  # [rad]: full motor range (calibrated at startup)
GRIPPER_FINGER_STROKE = 0.096  # [m]: total finger travel (both fingers combined)
GRIPPER_TRANSMISSION_RATIO = GRIPPER_FINGER_STROKE / GRIPPER_MOTOR_STROKE

# Reflect DM4310 motor properties through the transmission to the linear joint.
(
  GRIPPER_ARMATURE,
  GRIPPER_VELOCITY_LIMIT,
  GRIPPER_EFFORT_LIMIT,
) = reflect_rotary_to_linear(
  armature_rotary=ARMATURE_DM_4310,
  velocity_limit_rotary=DM_4310.velocity_limit,
  effort_limit_rotary=DM_4310.effort_limit,
  transmission_ratio=GRIPPER_TRANSMISSION_RATIO,
)

# PD gains: kp = M_eff * omega^2, kd = 2 * zeta * M_eff * omega.
NATURAL_FREQ_GRIPPER = 10.0 * 2.0 * 3.1415926535  # 2 Hz
STIFFNESS_GRIPPER = EFFECTIVE_INERTIAS["left_finger"] * NATURAL_FREQ_GRIPPER**2
DAMPING_GRIPPER = (
  2.0 * DAMPING_RATIO * EFFECTIVE_INERTIAS["left_finger"] * NATURAL_FREQ_GRIPPER
)

# Limit gripper force to 10% of motor capacity for sim stability.
# GRIPPER_EFFORT_LIMIT_SAFE = GRIPPER_EFFORT_LIMIT * 0.1

# Only actuate left_finger; right_finger is coupled via equality constraint.
GRIPPER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("left_finger",),
  stiffness=STIFFNESS_GRIPPER,
  damping=DAMPING_GRIPPER,
  effort_limit=GRIPPER_EFFORT_LIMIT,
  armature=GRIPPER_ARMATURE,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.01),
  joint_pos={
    "joint2": 1.047,
    "joint3": 1.05,
    "joint4": -0.9,
    "left_finger": 0.0475 / 2,
    "right_finger": 0.0475 / 2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={
    "tip_[lr]_\\d+_collision": 6,
    ".*_collision": 3,
  },
  friction={
    "tip_[lr]_\\d+_collision": (1, 5e-3, 5e-4),
    ".*_collision": (0.6,),
  },
  solref={
    "tip_[lr]_\\d+_collision": (0.01, 1),
  },
  priority={
    "tip_[lr]_\\d+_collision": 1,
  },
)

GRIPPER_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype={
    "(link6|gripper|tip)_.*_collision": 1,
    ".*_collision": 0,
  },
  conaffinity={
    "(link6|gripper|tip)_.*_collision": 1,
    ".*_collision": 0,
  },
  condim={
    "tip_[lr]_\\d+_collision": 6,
    ".*_collision": 3,
  },
  friction={
    "tip_[lr]_\\d+_collision": (1, 5e-3, 5e-4),
    ".*_collision": (0.6,),
  },
  solref={
    "tip_[lr]_\\d+_collision": (0.01, 1),
  },
  priority={
    "tip_[lr]_\\d+_collision": 1,
  },
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(*ARM_ACTUATORS, GRIPPER_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)


def get_yam_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(GRIPPER_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


YAM_ACTION_SCALE: dict[str, float] = {}
for a in ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    YAM_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import sys

  import numpy as np

  if len(sys.argv) > 1 and sys.argv[1] == "inertias":
    # Recompute effective inertias from the mass matrix at qpos0.
    # Usage: python yam_constants.py inertias
    spec = get_spec()
    model = spec.compile()

    joint_armatures = {
      "joint1": ARMATURE_DM_4340,
      "joint2": ARMATURE_DM_4340,
      "joint3": ARMATURE_DM_4340,
      "joint4": ARMATURE_DM_4310,
      "joint5": ARMATURE_DM_4310,
      "joint6": ARMATURE_DM_4310,
      "left_finger": GRIPPER_ARMATURE,
    }
    for name, arm in joint_armatures.items():
      jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      model.dof_armature[model.jnt_dofadr[jnt_id]] = arm

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)

    print("EFFECTIVE_INERTIAS = {")
    for name in joint_armatures:
      jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      dof = model.jnt_dofadr[jnt_id]
      print(f'  "{name}": {M[dof, dof]:.6f},')
    print("}")
  elif len(sys.argv) > 1 and sys.argv[1] == "gains":
    # Print PD gains for all actuators.
    # Usage: python yam_constants.py gains
    print(f"{'joint':<14} {'freq_hz':>8} {'kp':>10} {'kd':>10} {'kd_hw':>8}")
    print("-" * 54)
    for a in ARTICULATION.actuators:
      assert isinstance(a, BuiltinPositionActuatorCfg)
      name = a.target_names_expr[0]
      kp = a.stiffness
      kd = a.damping
      kd_hw = kd / TORQUE_CONSTANT_CORRECTION
      m = EFFECTIVE_INERTIAS[name]
      omega = kd / (2.0 * DAMPING_RATIO * m)
      freq = omega / (2.0 * 3.1415926535)
      print(f"{name:<14} {freq:>8.2f} {kp:>10.2f} {kd:>10.3f} {kd_hw:>8.3f}")
  else:
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_yam_robot_cfg())
    viewer.launch(robot.spec.compile())
