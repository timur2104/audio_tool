import argparse
import os.path

import numpy as np
import wave
from scipy import signal
import speech_recognition as sr
import json


def read_audio(audio_file: str) -> (int, int, int, bytes):
    """
    Function to read the audio file
    :param audio_file: path to the audio file
    :return: parameters of the audio file (framerate, nchannels, sampwidth) and the audio bytes (frames)
    """
    with wave.open(audio_file, 'rb') as wf:
        framerate = wf.getframerate()
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    return framerate, nchannels, sampwidth, frames


def write_audio(audio_file: str, framerate: int, nchannels: int, sampwidth: int, audio_bytes) -> None:
    """
    Function to write the audio file
    :param audio_file: path to the audio file
    :param framerate: framerate of the audio file
    :param nchannels: number of channels of the audio file
    :param sampwidth: sample width of the audio file
    :param audio_bytes: audio bytes of the audio file
    :return: None
    """
    with wave.open(audio_file, 'wb') as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(audio_bytes)


def bytes_to_samples(audio_data: bytes, sampwidth: int) -> np.ndarray:
    """
    Function to convert bytes to samples with respect to given sample width
    :param audio_data: bytes of the audio file
    :param sampwidth: sample width of the audio file
    :return: samples of samp_width bytes
    """
    # calculating the number of samples
    num_samples = len(audio_data) // sampwidth

    # Creating samples array
    samples = np.zeros(num_samples, dtype=np.int32)
    for i in range(num_samples):
        sample_bytes = audio_data[i*sampwidth:(i+1)*sampwidth]
        sample = int.from_bytes(sample_bytes, byteorder='little', signed=True)
        samples[i] = sample

    return samples


def samples_to_bytes(samples, sampwidth):
    """
    Function to convert samples to bytes with respect to given sample width
    :param samples: samples of samp_width bytes
    :param sampwidth: sample width of the audio file
    :return: bytes of the audio file
    """
    output_bytes = bytearray()
    for sample in samples:
        sample_bytes = int(sample).to_bytes(sampwidth, byteorder='little', signed=True)
        output_bytes.extend(sample_bytes)
    return output_bytes


def modify_audio_speed(audio_file: str, speed_factor: float, method: str = "fourier") -> None:
    """
    Function to modify the speed of the audio file with one of three methods:
        - fourier: using Fourier resampling method
        - polyphase: using polyphase filtering method for resampling
        - framerate: just changing the framerate
    :param audio_file: path to the audio file of .wav format
    :param speed_factor: factor by which the speed of the audio file is to be modified
    :param method: method to be used for speed modification
    :return: None
    """
    assert method == "fourier" or method == "polyphase" or method == "framerate", "Invalid method"

    # Read file by bytes (int8), because there are plenty of encodings with different sizes
    framerate, nchannels, sampwidth, frames = read_audio(audio_file)
    audio_bytes = np.frombuffer(frames, dtype=np.int8)

    out_filename = f"data/outputs/{os.path.basename(audio_file)[:-4]}_speed_{speed_factor}_{method}.wav"

    if method == "framerate":
        return write_audio(out_filename, framerate * speed_factor, nchannels, sampwidth, frames)

    new_sample_rate = int(framerate * speed_factor)
    if method == "fourier":
        modified_samples = signal.resample(audio_bytes, int(len(audio_bytes) * (new_sample_rate / framerate)))
        out_bytes = samples_to_bytes(modified_samples, sampwidth)
        return write_audio(out_filename, framerate, nchannels, sampwidth, out_bytes)
    else:
        modified_samples = signal.resample_poly(audio_bytes.astype(np.float32), new_sample_rate, framerate)
        out_bytes = samples_to_bytes(modified_samples, sampwidth)
        return write_audio(out_filename, framerate, nchannels, sampwidth, out_bytes)


def modify_audio_volume(audio_file: str, volume_factor: float) -> None:
    """
    Function to modify the volume of the audio file.
    Before multiplying the signal with a volume factor, normalization is being applied.
    :param audio_file: path to the audio file of .wav format
    :param volume_factor: factor by which the volume of the audio file is to be modified
    :return: None
    """
    # Read file by bytes (int8), because there are plenty of encodings with different sizes
    framerate, nchannels, sampwidth, frames = read_audio(audio_file)
    audio_bytes = np.frombuffer(frames, dtype=np.int8)

    # normalizing bytes to prevent exceeding the maximum and minimum values of int8
    # to avoid clipping and distortion
    max_amplitude = np.max(np.abs(audio_bytes))
    normalized_samples = audio_bytes.astype(np.float32) / max_amplitude

    # applying volume factor and clipping values
    scaled_samples = (normalized_samples * volume_factor).clip(-1.0, 1.0)

    # returning values to initial amplitude and dtype
    scaled_samples_int8 = (scaled_samples * max_amplitude).astype(np.int8)

    # converting back to bytes
    modified_audio_data = scaled_samples_int8.tobytes()

    # writing modified audio file
    write_audio(f"data/outputs/{os.path.basename(audio_file)[:-4]}_volume_{volume_factor}.wav",
                framerate, nchannels, sampwidth, modified_audio_data)


def transcribe_audio(audio_file: str, transcript_model: str) -> None:
    """
    Function to transcribe the audio file using the given transcript model.
    :param audio_file: path to the audio file of .wav format
    :param transcript_model: name of the transcript model
    :return: None
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        print(f'./data/model/{transcript_model}.pt')
        data = recognizer.recognize_whisper(audio, model=f'./data/model/{transcript_model}.pt', show_dict=True)
    with open(os.path.join("data/outputs", os.path.basename(audio_file)[:-4]) + f"_{transcript_model}.json", "w") as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='audio_tool',
                                     usage='python audio_tool.py [--speed <speed_factor> [--method <method>]]'
                                           ' [--volume <volume_factor>] '
                                           '[--transcript <model>] <filename>',
                                     description="Command line utility for working with audio files (.wav).\n"
                                                 "Three main operations are supported:\n"
                                                 "\t1. Modify audio speed\n"
                                                 "\t2. Modify audio volume\n"
                                                 "\t3. Add transcript to the audio file",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add arguments
    parser.add_argument("filename", help="Path to the audio file")
    parser.add_argument("--speed", type=float, help="Speed factor for modifying audio speed")
    parser.add_argument("--volume", type=float, help="Volume factor for modifying audio volume")
    parser.add_argument("--transcript", type=str, help="OpenAI model name for transcribing audio",
                        choices=['tiny', 'base'], default='base')
    parser.add_argument("--method", type=str,
                        choices=["fourier", "polyphase", "framerate"], default="fourier",)

    # Parse arguments
    args = parser.parse_args()

    assert args.method == "fourier" or args.method == "polyphase" or args.method == "framerate", "Invalid method"

    # Check which operation to perform
    if args.speed:
        modify_audio_speed(args.filename, args.speed, method=args.method)
    elif args.volume:
        modify_audio_volume(args.filename, args.volume)
    elif args.transcript:
        transcribe_audio(args.filename, args.transcript)
    else:
        print("Error: Specify either --speed, --volume or --transcript option.")
