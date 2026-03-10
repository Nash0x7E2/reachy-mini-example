"""
Reachy Mini Vision Agent

A local voice+vision agent for the Reachy Mini robot using the vision-agents SDK.
Uses Gemini VLM for multimodal understanding (camera + text), Cartesia for TTS,
Deepgram for STT, and the reachy_mini SDK for robot control.

Usage:
    uv run python reachy_mini_agent.py
"""

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
from vision_agents.core.edge.local_transport import LocalTransport
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


HEAD_ACTIONS: dict[str, tuple[float, float, float, float, float, float]] = {
    "nod": (0, 0, 0, 0, -20, 0),
    "shake": (0, 0, 0, 0, 0, 30),
    "look_left": (0, 0, 0, 0, 0, -30),
    "look_right": (0, 0, 0, 0, 0, 30),
    "look_up": (0, 0, 0, 0, -25, 0),
    "look_down": (0, 0, 0, 0, 25, 0),
    "curious": (0, 0, 0, 15, -10, 15),
}

EMOTION_ACTIONS: set[str] = {"happy", "sad", "surprised", "angry", "confused", "thinking"}


class ReachyController:
    """Manages connection to the Reachy Mini robot and dispatches actions."""

    def __init__(self):
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

            self._mini = ReachyMini()
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

        pitch = 3.0 * math.sin(self._idle_phase)
        antenna = 5.0 * math.sin(self._idle_phase * 0.7)
        self._mini.set_target(
            pitch=pitch,
            right_antenna=antenna,
            left_antenna=-antenna,
        )
        self._idle_phase += 0.04

    async def _play_emotion(self, emotion: str):
        """Play a recorded emotion on the robot."""
        if self._mini is None or self._moves is None:
            return
        move = self._moves.get(emotion)
        if move is None:
            logger.warning("Unknown emotion action: %s", emotion)
            return
        await asyncio.to_thread(self._mini.play_move, move, initial_goto_duration=0.5)

    async def _move_head(self, action: str):
        """Move the robot head to a target pose then return to neutral."""
        if self._mini is None:
            return
        target = HEAD_ACTIONS.get(action)
        if target is None:
            return
        await asyncio.to_thread(
            self._mini.goto_target,
            x=target[0],
            y=target[1],
            z=target[2],
            roll=target[3],
            pitch=target[4],
            yaw=target[5],
            duration=0.5,
        )
        await asyncio.sleep(0.6)
        await asyncio.to_thread(
            self._mini.goto_target,
            x=0,
            y=0,
            z=0,
            roll=0,
            pitch=0,
            yaw=0,
            duration=0.5,
        )

    async def _wiggle_antennas(self):
        """Quick antenna wiggle gesture."""
        if self._mini is None:
            return
        for angle in [30, -30, 20, -20, 0]:
            await asyncio.to_thread(
                self._mini.goto_target,
                right_antenna=float(angle),
                left_antenna=float(-angle),
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
) -> tuple[Agent, ReachyController]:
    """Create the vision agent and Reachy controller."""
    vlm = gemini.VLM(model="gemini-2.5-flash-preview-05-20")

    input_sample_rate = get_device_sample_rate(input_device, is_input=True)
    output_sample_rate = get_device_sample_rate(output_device, is_input=False)

    logger.info("Input sample rate: %dHz", input_sample_rate)
    logger.info("Output sample rate: %dHz", output_sample_rate)
    if video_device:
        logger.info("Video device: %s", video_device)

    transport = LocalTransport(
        sample_rate=output_sample_rate,
        input_device=input_device,
        output_device=output_device,
        video_device=video_device,
    )

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

    controller = ReachyController()
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
):
    """Run the Reachy Mini vision agent."""
    agent, controller = await create_agent(input_device, output_device, video_device)

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


def main():
    print("\n" + "=" * 60)
    print("  Reachy Mini Vision Agent")
    print("  Gemini VLM + Cartesia TTS + AssemblyAI STT")
    print("=" * 60)
    print("\nThis agent uses your local microphone, speakers, and camera.")
    print("The robot (if connected) will react during conversation.\n")

    input_device, output_device = select_audio_devices()
    video_device = select_video_device()

    print("Speak into your microphone to interact with Reachy Mini.")
    if video_device:
        print("Camera is enabled — Reachy can see!")
    print("Press Ctrl+C to stop.\n")

    try:
        asyncio.run(run_agent(input_device, output_device, video_device))
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
