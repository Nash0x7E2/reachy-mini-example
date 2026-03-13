"""
Reachy Mini audio adapters for Vision Agents LocalTransport.

Wraps the Reachy Mini SDK's GStreamerAudio (appsink/appsrc pipelines) into
the AudioInput / AudioOutput protocols expected by LocalTransport, allowing
the robot's onboard ReSpeaker mic array and speaker to be used as the
agent's audio I/O.

Usage::

    from reachy_mini.media.audio_gstreamer import GStreamerAudio
    from reachy_audio import ReachyAudioInput, ReachyAudioOutput

    gst = GStreamerAudio()
    transport = LocalTransport(
        audio_input=ReachyAudioInput(gst),
        audio_output=ReachyAudioOutput(gst),
    )
"""

import numpy as np
from reachy_mini.media.audio_base import AudioBase


class ReachyAudioInput:
    """AudioInput adapter wrapping Reachy Mini's GStreamer mic pipeline.

    Reads stereo float32 frames from the appsink and returns mono int16
    samples suitable for Vision Agents' PcmData.
    """

    def __init__(self, audio: AudioBase):
        self._audio = audio

    @property
    def sample_rate(self) -> int:
        return self._audio.get_input_audio_samplerate()

    @property
    def channels(self) -> int:
        return 1

    def start(self) -> None:
        self._audio.start_recording()

    def read(self) -> np.ndarray | None:
        """Blocking read from the GStreamer appsink (up to ~20ms timeout).

        Returns flat int16 mono samples, or None on timeout.
        """
        sample = self._audio.get_audio_sample()
        if sample is None:
            return None
        mono = sample.mean(axis=1)
        return (mono * 32767).astype(np.int16)

    def stop(self) -> None:
        self._audio.stop_recording()


class ReachyAudioOutput:
    """AudioOutput adapter wrapping Reachy Mini's GStreamer speaker pipeline.

    Accepts flat int16 samples from LocalOutputAudioTrack and converts
    them to the stereo float32 format expected by the appsrc.
    """

    def __init__(self, audio: AudioBase):
        self._audio = audio

    @property
    def sample_rate(self) -> int:
        return self._audio.get_output_audio_samplerate()

    @property
    def channels(self) -> int:
        return self._audio.get_output_channels()

    def start(self) -> None:
        self._audio.start_playing()

    def write(self, samples: np.ndarray) -> None:
        """Convert int16 to float32 stereo and push to the GStreamer appsrc."""
        float_samples = samples.astype(np.float32) / 32767.0
        output_channels = self._audio.get_output_channels()
        frames = len(float_samples) // output_channels
        if output_channels > 1 and float_samples.ndim == 1:
            if len(float_samples) % output_channels == 0:
                float_samples = float_samples.reshape(frames, output_channels)
            else:
                float_samples = np.column_stack(
                    [float_samples] * output_channels
                )
        self._audio.push_audio_sample(float_samples)

    def stop(self) -> None:
        self._audio.stop_playing()
