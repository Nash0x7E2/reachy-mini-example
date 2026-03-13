"""
Reachy Mini Vision Agent

A local voice+vision agent for the Reachy Mini robot using the vision-agents SDK.
Uses Gemini VLM for multimodal understanding (camera + text), Cartesia for TTS,
Deepgram for STT, and the reachy_mini SDK for robot control.

Usage:
    uv run python reachy_mini_agent.py
"""

import argparse
import asyncio
import logging
import math
import re
import signal
import sys
from typing import AsyncIterator, Iterator, Optional, Union

import numpy as np
from dotenv import load_dotenv
from getstream.video.rtc import PcmData
from getstream.video.rtc.track_util import AudioFormat

from vision_agents.core import Agent, User
from vision_agents.core.edge.local_devices import (
    get_device_sample_rate,
    select_audio_devices,
    select_video_device,
)
from vision_agents.core.edge.local_transport import AudioInput, AudioOutput, LocalTransport
from vision_agents.core.llm.events import LLMResponseCompletedEvent
from vision_agents.core.tts.tts import TTS
from vision_agents.plugins import assemblyai, cartesia, gemini

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

INSTRUCTIONS = """\
You are Reachy Mini, a small expressive robot made by Pollen Robotics.
You can see through your camera and hear through the microphone.

Personality:
- Curious and playful, you love observing the world around you.
- Keep responses short (1-3 sentences) and conversational.
- Don't use special characters, markdown, or formatting — your words are spoken aloud.
- When you see something interesting through the camera, comment on it naturally.

When responding, you may optionally include a single action tag on its own line at the
very end of your response. The tag must be exactly one of:

  [happy] [sad] [surprised] [curious] [angry] [confused] [thinking] [nod] [shake] [look_left] [look_right] [look_up] [look_down] [wiggle_antennas]

Only include a tag when the emotion or gesture genuinely fits your response.
Do not include a tag in every response — only when it adds expressiveness.
Never include the tag inside a sentence; always place it on its own line at the end.
"""

ACTION_PATTERN = re.compile(
    r"\[(happy|sad|surprised|curious|angry|confused|thinking"
    r"|nod|shake|look_left|look_right|look_up|look_down|wiggle_antennas)\]"
)
ACTION_LINE_PATTERN = re.compile(
    r"^\[(happy|sad|surprised|curious|angry|confused|thinking"
    r"|nod|shake|look_left|look_right|look_up|look_down|wiggle_antennas)\]$"
)


class ActionStrippingTTS(TTS):
    """TTS wrapper that strips robot action tags before synthesising speech."""

    def __init__(self, inner: TTS):
        super().__init__(provider_name=inner.provider_name)
        self._inner = inner
        self.events = inner.events

    async def stream_audio(
        self, text: str, *args, **kwargs
    ) -> Union[PcmData, Iterator[PcmData], AsyncIterator[PcmData]]:
        cleaned = ACTION_PATTERN.sub("", text).strip()
        if not cleaned:
            return PcmData(
                samples=np.zeros(1, dtype=np.int16),
                sample_rate=16000,
                format=AudioFormat.S16,
                channels=1,
            )
        return await self._inner.stream_audio(cleaned, *args, **kwargs)

    async def stop_audio(self) -> None:
        await self._inner.stop_audio()


HEAD_ACTIONS: dict[str, tuple[float, float, float]] = {
    "nod": (0, -20, 0),
    "shake": (0, 0, 30),
    "look_left": (0, 0, -30),
    "look_right": (0, 0, 30),
    "look_up": (-25, 0, 0),
    "look_down": (25, 0, 0),
    "curious": (15, -10, 15),
}


def _rpy_to_pose(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build a 4x4 pose matrix from roll/pitch/yaw in degrees."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0],
        [-sp,     cp * sr,                cp * cr,                0],
        [0,       0,                      0,                      1],
    ], dtype=np.float64)

EMOTION_TO_MOVE: dict[str, str] = {
    "happy": "cheerful1",
    "sad": "sad1",
    "surprised": "surprised1",
    "angry": "furious1",
    "confused": "confused1",
    "thinking": "thoughtful1",
}
EMOTION_ACTIONS: set[str] = set(EMOTION_TO_MOVE)


class ReachyController:
    """Manages connection to the Reachy Mini robot and dispatches actions."""

    def __init__(self, host: Optional[str] = None):
        self._host = host
        self._mini = None
        self._moves = None
        self._action_queue: asyncio.Queue[str] = asyncio.Queue()
        self._control_task: Optional[asyncio.Task] = None
        self._running = False
        self._idle_phase = 0.0
        self._control_period_s = 0.02  # 50Hz

    async def connect(self):
        """Connect to the robot and load the emotions library."""
        try:
            from reachy_mini import ReachyMini
            from reachy_mini.motion.recorded_move import RecordedMoves

            kwargs = {}
            if self._host is not None:
                kwargs["host"] = self._host
                kwargs["connection_mode"] = "network"
            self._mini = ReachyMini(**kwargs)
            await asyncio.to_thread(self._mini.__enter__)

            self._moves = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
            logger.info("Connected to Reachy Mini")
        except ImportError:
            logger.warning(
                "reachy_mini not installed or daemon not running — "
                "robot actions will be logged but not executed"
            )
        except TimeoutError:
            logger.warning(
                "Could not connect to Reachy Mini daemon — "
                "robot actions will be logged but not executed"
            )

    async def disconnect(self):
        await self.stop_control_loop()

        if self._mini is not None:
            try:
                await asyncio.to_thread(self._mini.__exit__, None, None, None)
            except (RuntimeError, OSError):
                logger.exception("Error disconnecting from Reachy Mini")
            self._mini = None

    async def enqueue_action(self, action: str):
        """Queue a parsed action tag for serialized execution."""
        if self._mini is None:
            logger.info("Skipping action %s because robot is unavailable", action)
            return
        logger.info("Queueing robot action: %s", action)
        await self._action_queue.put(action)

    def start_control_loop(self):
        """Start the controller loop as the only set_target owner."""
        if self._mini is None:
            return
        if self._control_task is not None and not self._control_task.done():
            return
        self._running = True
        self._control_task = asyncio.create_task(self._control_loop())

    async def stop_control_loop(self):
        """Stop the controller loop and wait for shutdown."""
        self._running = False
        if self._control_task is None:
            return
        self._control_task.cancel()
        try:
            await self._control_task
        except asyncio.CancelledError:
            pass
        self._control_task = None

    async def _control_loop(self):
        """Main controller loop. This is the only place that calls set_target()."""
        try:
            loop = asyncio.get_running_loop()
            while self._running:
                tick_started = loop.time()
                try:
                    await self._run_next_action()
                    self._apply_idle_target()
                except (RuntimeError, ValueError):
                    logger.exception("Control loop action failed")

                elapsed = loop.time() - tick_started
                sleep_s = self._control_period_s - elapsed
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
        except asyncio.CancelledError:
            raise

    async def _run_next_action(self):
        """Run at most one queued action per control tick."""
        if self._mini is None:
            return

        try:
            action = self._action_queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        if action in EMOTION_ACTIONS and self._moves is not None:
            await self._play_emotion(action)
        elif action in HEAD_ACTIONS:
            await self._move_head(action)
        elif action == "wiggle_antennas":
            await self._wiggle_antennas()
        else:
            logger.warning("Ignoring unsupported action: %s", action)

    def _apply_idle_target(self):
        """Apply subtle breathing-like motion while idle."""
        if self._mini is None:
            return

        pitch_deg = 3.0 * math.sin(self._idle_phase)
        antenna_rad = math.radians(5.0 * math.sin(self._idle_phase * 0.7))
        self._mini.set_target(
            head=_rpy_to_pose(0, pitch_deg, 0),
            antennas=[antenna_rad, -antenna_rad],
        )
        self._idle_phase += 0.04

    async def _play_emotion(self, emotion: str):
        """Play a recorded emotion on the robot."""
        if self._mini is None or self._moves is None:
            return
        move_name = EMOTION_TO_MOVE.get(emotion)
        if move_name is None:
            logger.warning("No move mapping for emotion: %s", emotion)
            return
        move = self._moves.get(move_name)
        await asyncio.to_thread(self._mini.play_move, move, initial_goto_duration=0.5)

    async def _move_head(self, action: str):
        """Move the robot head to a target pose then return to neutral."""
        if self._mini is None:
            return
        target = HEAD_ACTIONS.get(action)
        if target is None:
            return
        roll_deg, pitch_deg, yaw_deg = target
        pose = _rpy_to_pose(roll_deg, pitch_deg, yaw_deg)
        await asyncio.to_thread(
            self._mini.goto_target, head=pose, duration=0.5,
        )
        await asyncio.sleep(0.6)
        await asyncio.to_thread(
            self._mini.goto_target, head=np.eye(4), duration=0.5,
        )

    async def _wiggle_antennas(self):
        """Quick antenna wiggle gesture."""
        if self._mini is None:
            return
        for angle_deg in [30, -30, 20, -20, 0]:
            rad = math.radians(angle_deg)
            await asyncio.to_thread(
                self._mini.goto_target,
                antennas=[rad, -rad],
                duration=0.15,
            )
            await asyncio.sleep(0.18)

    async def execute_action(self, action: str):
        """Backward-compatible action entry point."""
        await self.enqueue_action(action)


def extract_action(text: str) -> Optional[str]:
    """Extract action tag only when it is the last non-empty line."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    match = ACTION_LINE_PATTERN.fullmatch(lines[-1])
    if match is None:
        return None
    return match.group(1)


async def create_agent(
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    video_device: Optional[str] = None,
    audio_input: Optional[AudioInput] = None,
    audio_output: Optional[AudioOutput] = None,
    robot_host: Optional[str] = None,
) -> tuple[Agent, ReachyController]:
    """Create the vision agent and Reachy controller."""
    vlm = gemini.VLM(model="gemini-2.5-flash")

    if audio_input is not None and audio_output is not None:
        logger.info("Using custom audio backends (robot audio)")
        logger.info(
            "Input: %dHz %dch, Output: %dHz %dch",
            audio_input.sample_rate,
            audio_input.channels,
            audio_output.sample_rate,
            audio_output.channels,
        )
        transport = LocalTransport(
            audio_input=audio_input,
            audio_output=audio_output,
            video_device=video_device,
        )
    else:
        input_sample_rate = get_device_sample_rate(input_device, is_input=True)
        output_sample_rate = get_device_sample_rate(output_device, is_input=False)
        logger.info("Input sample rate: %dHz", input_sample_rate)
        logger.info("Output sample rate: %dHz", output_sample_rate)
        transport = LocalTransport(
            sample_rate=output_sample_rate,
            input_device=input_device,
            output_device=output_device,
            video_device=video_device,
        )

    if video_device:
        logger.info("Video device: %s", video_device)

    tts = ActionStrippingTTS(cartesia.TTS(model_id="sonic-3"))

    agent = Agent(
        edge=transport,
        agent_user=User(name="Reachy Mini", id="reachy-mini-agent"),
        instructions=INSTRUCTIONS,
        processors=[],
        llm=vlm,
        tts=tts,
        stt=assemblyai.STT(),
        streaming_tts=True,
    )

    controller = ReachyController(host=robot_host)
    await controller.connect()

    @agent.subscribe
    async def on_response(event: LLMResponseCompletedEvent):
        if not event.text:
            return
        action = extract_action(event.text)
        if action:
            await controller.enqueue_action(action)

    return agent, controller


async def run_agent(
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    video_device: Optional[str] = None,
    audio_input: Optional[AudioInput] = None,
    audio_output: Optional[AudioOutput] = None,
    robot_host: Optional[str] = None,
):
    """Run the Reachy Mini vision agent."""
    agent, controller = await create_agent(
        input_device, output_device, video_device, audio_input, audio_output, robot_host
    )

    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        logger.info("Starting Reachy Mini agent...")
        logger.info("Speak into your microphone. Press Ctrl+C to stop.")
        logger.info(
            "Testing tip: start sim with `reachy-mini-daemon --sim` if hardware is unavailable."
        )

        async with agent.join(call=None, participant_wait_timeout=0):
            controller.start_control_loop()
            await agent.simple_response("Greet the user, introduce yourself as Reachy Mini")
            await shutdown_event.wait()

    except asyncio.CancelledError:
        logger.info("Agent task cancelled")
    finally:
        logger.info("Shutting down...")
        await controller.disconnect()
        await agent.close()
        logger.info("Reachy Mini agent stopped")


def _create_robot_audio() -> tuple[AudioInput, AudioOutput]:
    """Create Reachy Mini GStreamer audio backends."""
    from reachy_mini.media.audio_gstreamer import GStreamerAudio

    from reachy_audio import ReachyAudioInput, ReachyAudioOutput

    gst = GStreamerAudio()
    return ReachyAudioInput(gst), ReachyAudioOutput(gst)


def main():
    parser = argparse.ArgumentParser(description="Reachy Mini Vision Agent")
    parser.add_argument(
        "--robot-audio",
        action="store_true",
        help="Use the robot's onboard mic and speaker (GStreamer) instead of laptop audio",
    )
    parser.add_argument(
        "--robot-host",
        type=str,
        default=None,
        help="IP or hostname of the Reachy Mini daemon (e.g. 192.168.128.120)",
    )
    parser.add_argument(
        "--default-audio",
        action="store_true",
        help="Use system default mic and speakers (skip interactive device selection)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable camera",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Reachy Mini Vision Agent")
    print("  Gemini VLM + Cartesia TTS + AssemblyAI STT")
    print("=" * 60)

    if args.robot_host:
        print(f"\nConnecting to robot at {args.robot_host}")

    audio_in: Optional[AudioInput] = None
    audio_out: Optional[AudioOutput] = None
    input_device: Optional[int] = None
    output_device: Optional[int] = None

    if args.robot_audio:
        print("\nUsing Reachy Mini onboard mic and speaker (GStreamer).\n")
        audio_in, audio_out = _create_robot_audio()
    elif args.default_audio:
        print("\nUsing system default mic and speakers.\n")
    else:
        print("\nThis agent uses your local microphone, speakers, and camera.")
        print("The robot (if connected) will react during conversation.\n")
        input_device, output_device = select_audio_devices()

    video_device: Optional[str] = None
    if args.no_video:
        print("Camera disabled.")
    else:
        video_device = select_video_device()

    print("Speak into your microphone to interact with Reachy Mini.")
    if video_device:
        print("Camera is enabled — Reachy can see!")
    print("Press Ctrl+C to stop.\n")

    try:
        asyncio.run(
            run_agent(
                input_device, output_device, video_device,
                audio_in, audio_out, args.robot_host,
            )
        )
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
