"""Module which defines functions for formant shifting audio."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lazy_loader as lazy

from pathlib import Path

from ultimate_rvc.core.common import (
    get_file_hash,
    json_dump,
    validate_audio_file_exists,
)
from ultimate_rvc.core.exceptions import Entity
from ultimate_rvc.core.generate.common import get_unique_base_path
from ultimate_rvc.core.generate.typing_extra import (
    FileMetaData,
    FormantShiftMetaData,
)

if TYPE_CHECKING:
    import numpy as np

    import librosa
    import parselmouth
    import soundfile as sf

    from ultimate_rvc.typing_extra import StrPath
else:
    librosa = lazy.load("librosa")
    np = lazy.load("numpy")
    parselmouth = lazy.load("parselmouth")
    sf = lazy.load("soundfile")


def _apply_formant_shift(
    audio_path: Path,
    output_path: Path,
    formant_shift_ratio: float,
    pitch_range_factor: float,
) -> None:
    """
    Apply formant shifting to an audio file using Parselmouth.

    Parameters
    ----------
    audio_path : Path
        The path to the input audio file.
    output_path : Path
        The path to write the formant-shifted audio file.
    formant_shift_ratio : float
        The ratio for shifting vocal formants.
    pitch_range_factor : float
        The factor for scaling the pitch variation range.

    """
    y, _sr = librosa.load(str(audio_path), sr=16000, mono=True)
    snd = parselmouth.Sound(y, sampling_frequency=16000)
    shifted = parselmouth.praat.call(
        snd,
        "Change gender",
        75.0,
        600.0,
        formant_shift_ratio,
        0.0,
        pitch_range_factor,
        1.0,
    )
    out_array = shifted.values[0]
    sf.write(str(output_path), out_array, 16000)


def formant_shift(
    audio_track: StrPath,
    directory: StrPath,
    formant_shift_ratio: float = 1.0,
    pitch_range_factor: float = 1.0,
) -> Path:
    """
    Apply formant shifting to an audio track.

    If both the formant shift ratio and pitch range factor are 1.0,
    the original audio track is returned unchanged.

    Parameters
    ----------
    audio_track : StrPath
        The path to the audio track to apply formant shifting to.
    directory : StrPath
        The path to the directory where the formant-shifted audio
        track will be saved.
    formant_shift_ratio : float, default=1.0
        The ratio for shifting vocal formants. Values greater than
        1.0 raise formants, values less than 1.0 lower them.
    pitch_range_factor : float, default=1.0
        The factor for scaling the pitch variation range.

    Returns
    -------
    Path
        The path to the formant-shifted audio track, or the original
        audio track if no shifting was applied.

    """
    audio_path = validate_audio_file_exists(audio_track, Entity.AUDIO_TRACK)

    if formant_shift_ratio == 1.0 and pitch_range_factor == 1.0:  # noqa: RUF069
        return audio_path

    directory_path = Path(directory)

    args_dict = FormantShiftMetaData(
        audio_track=FileMetaData(
            name=audio_path.name,
            hash_id=get_file_hash(audio_path),
        ),
        formant_shift_ratio=formant_shift_ratio,
        pitch_range_factor=pitch_range_factor,
    ).model_dump()

    paths = [
        get_unique_base_path(
            directory_path,
            "20b_Formant_Shifted",
            args_dict,
        ).with_suffix(suffix)
        for suffix in [".wav", ".json"]
    ]

    shifted_path, shifted_json_path = paths

    if not all(path.exists() for path in paths):
        _apply_formant_shift(
            audio_path,
            shifted_path,
            formant_shift_ratio,
            pitch_range_factor,
        )
        json_dump(args_dict, shifted_json_path)

    return shifted_path
