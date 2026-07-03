"""
 ___________________
< author:pablo_feng >
 -------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
Joystick interactive play for velocity policies in unitree_rl_mjlab.

Usage:
    python scripts/play_joystick_control.py Unitree-G1-Flat \
        --viewer native \
        --checkpoint_file logs/rsl_rl/g1_velocity/2026-03-11_15-48-18/model_4100.pt

Default joystick mapping:
    left stick up/down   -> vx
    left stick left/right-> vy
    right stick left/right -> wz

Optional buttons:
    A / Cross -> zero command
    B / Circle -> toggle manual override on/off
"""

import os
import sys
import time
import types
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pygame
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = False

    # joystick teleop settings
    interactive: bool = True
    lin_x_cmd: float = 0.6
    lin_y_cmd: float = 0.4
    ang_z_cmd: float = 1.0
    deadzone: float = 0.15
    joystick_id: int = 0

    # common default mapping on many controllers
    axis_lx: int = 0
    axis_ly: int = 1
    axis_rx: int = 3

    # button mapping (set to -1 to disable)
    button_zero: int = 0
    button_toggle: int = 1


def _resolve_viewer(viewer: str) -> str:
    if viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        return "native" if has_display else "viser"
    return viewer


def _is_velocity_term(term) -> bool:
    cfg = getattr(term, "cfg", None)
    ranges = getattr(cfg, "ranges", None)
    has_ranges = (
        ranges is not None
        and hasattr(ranges, "lin_vel_x")
        and hasattr(ranges, "lin_vel_y")
        and hasattr(ranges, "ang_vel_z")
    )
    has_storage = hasattr(term, "vel_command_b")
    has_updater = hasattr(term, "_update_command")
    return has_ranges and has_storage and has_updater


def _patch_velocity_term(term) -> None:
    """Patch a velocity command term so manual override is applied last."""
    if getattr(term, "_interactive_patch_installed", False):
        return

    if hasattr(term, "set_manual_override") and hasattr(term, "clear_manual_override"):
        term._interactive_patch_installed = True
        return

    term._manual_vel = None
    term._interactive_original_update_command = term._update_command

    def set_manual_override(self, lin_x: float, lin_y: float, ang_z: float) -> None:
        self._manual_vel = torch.tensor(
            [lin_x, lin_y, ang_z],
            device=self.vel_command_b.device,
            dtype=self.vel_command_b.dtype,
        )

    def clear_manual_override(self) -> None:
        self._manual_vel = None

    def patched_update_command(self, *args, **kwargs):
        self._interactive_original_update_command(*args, **kwargs)
        if self._manual_vel is not None:
            self.vel_command_b[:] = self._manual_vel

    term.set_manual_override = types.MethodType(set_manual_override, term)
    term.clear_manual_override = types.MethodType(clear_manual_override, term)
    term._update_command = types.MethodType(patched_update_command, term)
    term._interactive_patch_installed = True


class SharedJoystickState:
    def __init__(self):
        self.enabled = True
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self._lock = threading.Lock()

    def set_cmd(self, vx: float, vy: float, wz: float) -> None:
        with self._lock:
            self.vx = float(vx)
            self.vy = float(vy)
            self.wz = float(wz)

    def get_cmd(self):
        with self._lock:
            return self.vx, self.vy, self.wz

    def zero(self) -> None:
        self.set_cmd(0.0, 0.0, 0.0)

    def toggle(self) -> None:
        with self._lock:
            self.enabled = not self.enabled

    def is_enabled(self) -> bool:
        with self._lock:
            return self.enabled


class GamepadReader:
    def __init__(self, cfg: PlayConfig, state: SharedJoystickState):
        self.cfg = cfg
        self.state = state
        self.running = True

        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count <= cfg.joystick_id:
            raise RuntimeError(
                f"No joystick with id={cfg.joystick_id}. "
                f"Detected {count} joystick(s)."
            )

        self.joystick = pygame.joystick.Joystick(cfg.joystick_id)
        self.joystick.init()

        self.prev_toggle_pressed = False
        self.prev_zero_pressed = False

        print(f"[INFO] Using joystick: {self.joystick.get_name()}", flush=True)
        print(f"[INFO] num_axes={self.joystick.get_numaxes()}, num_buttons={self.joystick.get_numbuttons()}", flush=True)

    def _dz(self, value: float) -> float:
        if abs(value) < self.cfg.deadzone:
            return 0.0
        return value

    def _get_axis_safe(self, axis_id: int) -> float:
        if axis_id < 0 or axis_id >= self.joystick.get_numaxes():
            return 0.0
        return self.joystick.get_axis(axis_id)

    def _get_button_safe(self, button_id: int) -> bool:
        if button_id < 0 or button_id >= self.joystick.get_numbuttons():
            return False
        return bool(self.joystick.get_button(button_id))

    def loop(self):
        while self.running:
            pygame.event.pump()

            lx = self._dz(self._get_axis_safe(self.cfg.axis_lx))
            ly = self._dz(self._get_axis_safe(self.cfg.axis_ly))
            rx = self._dz(self._get_axis_safe(self.cfg.axis_rx))

            # Common convention: pushing stick forward gives negative LY.
            vx = -ly * self.cfg.lin_x_cmd
            vy = -lx * self.cfg.lin_y_cmd
            wz = -rx * self.cfg.ang_z_cmd

            self.state.set_cmd(vx, vy, wz)

            zero_pressed = self._get_button_safe(self.cfg.button_zero)
            if zero_pressed and not self.prev_zero_pressed:
                self.state.zero()
                print("[JOYSTICK] zero command", flush=True)
            self.prev_zero_pressed = zero_pressed

            toggle_pressed = self._get_button_safe(self.cfg.button_toggle)
            if toggle_pressed and not self.prev_toggle_pressed:
                self.state.toggle()
                print(f"[JOYSTICK] manual override = {self.state.is_enabled()}", flush=True)
            self.prev_toggle_pressed = toggle_pressed

            time.sleep(0.01)

    def stop(self):
        self.running = False


class JoystickVelocityController:
    def __init__(self, vel_terms: list, cfg: PlayConfig):
        self.vel_terms = vel_terms
        self.cfg = cfg
        self.state = SharedJoystickState()
        self.reader = GamepadReader(cfg, self.state)
        self.thread = threading.Thread(target=self.reader.loop, daemon=True)

        # Clamp default max commands to task ranges if present.
        first_cfg = getattr(self.vel_terms[0], "cfg", None)
        ranges = getattr(first_cfg, "ranges", None)
        if ranges is not None:
            print(
                "[INFO] Task command ranges: "
                f"vx={tuple(ranges.lin_vel_x)}, "
                f"vy={tuple(ranges.lin_vel_y)}, "
                f"wz={tuple(ranges.ang_vel_z)}",
                flush=True,
            )

    def start(self):
        self.thread.start()
        print("[INFO] Joystick reader started", flush=True)

    def stop(self):
        self.reader.stop()

    def apply(self):
        if self.state.is_enabled():
            vx, vy, wz = self.state.get_cmd()
            for term in self.vel_terms:
                term.set_manual_override(vx, vy, wz)
        else:
            for term in self.vel_terms:
                term.clear_manual_override()


def _install_joystick_velocity_override(env, cfg: PlayConfig):
    manager = getattr(env.unwrapped, "command_manager", None)
    terms_dict = getattr(manager, "_terms", None)
    if not isinstance(terms_dict, dict):
        print("[WARN] command_manager._terms not found; joystick control disabled.", flush=True)
        return None

    names = []
    vel_terms = []
    for name, term in terms_dict.items():
        if _is_velocity_term(term):
            _patch_velocity_term(term)
            vel_terms.append(term)
            names.append(name)

    if not vel_terms:
        print("[WARN] No velocity command term found; joystick control disabled.", flush=True)
        return None

    print(f"[INFO] Interactive velocity terms: {names}", flush=True)
    controller = JoystickVelocityController(vel_terms, cfg)
    controller.start()
    return controller


class PolicyWithJoystickOverride:
    """Small wrapper to apply joystick command before every policy call."""

    def __init__(self, base_policy, controller):
        self.base_policy = base_policy
        self.controller = controller

    def __call__(self, obs):
        if self.controller is not None:
            self.controller.apply()
        return self.base_policy(obs)


def run_play(task_id: str, cfg: PlayConfig):
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved_viewer = _resolve_viewer(cfg.viewer)

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO] Terminations disabled", flush=True)

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    controller = None
    if cfg.interactive:
        controller = _install_joystick_velocity_override(env, cfg)

    if cfg.agent in {"zero", "random"}:
        action_shape = env.unwrapped.action_space.shape

        class PolicyZero:
            def __call__(self, obs):
                del obs
                return torch.zeros(action_shape, device=env.unwrapped.device)

        class PolicyRandom:
            def __call__(self, obs):
                del obs
                return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

        base_policy = PolicyZero() if cfg.agent == "zero" else PolicyRandom()
    else:
        if cfg.checkpoint_file is None:
            raise ValueError("`--checkpoint_file` is required when agent='trained'.")

        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")

        print(f"[INFO] Loading checkpoint: {resume_path}", flush=True)

        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(
            str(resume_path),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        base_policy = runner.get_inference_policy(device=device)

    policy = PolicyWithJoystickOverride(base_policy, controller)

    print("\n=== Joystick teleop ===", flush=True)
    print("Left stick Y : vx", flush=True)
    print("Left stick X : vy", flush=True)
    print("Right stick X: wz", flush=True)
    print(f"deadzone={cfg.deadzone}", flush=True)
    print(f"button_zero={cfg.button_zero}, button_toggle={cfg.button_toggle}", flush=True)
    print("================================\n", flush=True)

    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    if controller is not None:
        controller.stop()
    env.close()


def main():
    import mjlab.tasks  # noqa: F401
    import mjlab

    all_tasks = list_tasks()

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    run_play(chosen_task, args)


if __name__ == "__main__":
    main()
