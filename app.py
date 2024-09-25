import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel

import voice_service as vs
from rag import get_rag_answer

# DEFAULT_MODEL_SIZE = "medium"
# DEFAULT_CHUNK_LENGTH = 10

# def is_silence(data, max_amplitude_threshold=3000):
#     """Check if audio data contains silence."""
#     max_amplitude = np.max(np.abs(data))
#     return max_amplitude <= max_amplitude_threshold


# def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
#     frames = []
#     for _ in range(0, int(16000 / 1024 * chunk_length)):
#         data = stream.read(1024)
#         frames.append(data)

#     temp_file_path = 'temp_audio_chunk.wav'
#     with wave.open(temp_file_path, 'wb') as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
#         wf.setframerate(16000)
#         wf.writeframes(b''.join(frames))

#     # Check if the recorded chunk contains silence
#     try:
#         samplerate, data = wavfile.read(temp_file_path)
#         if is_silence(data):
#             os.remove(temp_file_path)
#             return True
#         else:
#             return False
#     except Exception as e:
#         print(f"Error while reading audio file: {e}")
#         return False

    

# def transcribe_audio(model, file_path):
#     segments, info = model.transcribe(file_path, beam_size=7)
#     transcription = ' '.join(segment.text for segment in segments)
#     return transcription


# def main():
    
#     model_size = DEFAULT_MODEL_SIZE + ".en"
#     model = WhisperModel(model_size, device="cpu", compute_type="int8", num_workers=10)
    
#     audio = pyaudio.PyAudio()
#     stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
#     customer_input_transcription = ""

#     try:
#         while True:
#             chunk_file = "temp_audio_chunk.wav"
            
#             # Record audio chunk
#             print("_")
#             if not record_audio_chunk(audio, stream):
#                 # Transcribe audio
#                 transcription = transcribe_audio(model, chunk_file)
#                 os.remove(chunk_file)
#                 print("Customer:{}".format(transcription))
                
#                 # Add customer input to transcript
#                 customer_input_transcription += "Customer: " + transcription + "\n"
                
#                 # Process customer input and get response from AI assistant
#                 output = get_rag_answer(transcription)
#                 if output:
#                     output = output.lstrip()
#                     print("AI Assistant:{}".format(output))
#                     vs.play_text_to_speech(output)
                    
    
#     except KeyboardInterrupt:
#         print("\nStopping...")

#     finally:
#         stream.stop_stream()
#         stream.close()
#         audio.terminate()

# if __name__ == "__main__":
#     main()

import noisereduce as nr
import numpy as np
import pyaudio
import wave
import os
from scipy.io import wavfile

DEFAULT_MODEL_SIZE = "medium"
DEFAULT_CHUNK_LENGTH = 10

def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold


def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH, noise_profile=None):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        
        # Apply noise reduction if a noise profile is available
        if noise_profile is not None:
            reduced_data = nr.reduce_noise(y=data, sr=samplerate, y_noise=noise_profile)
        else:
            reduced_data = data
        
        # Check if the reduced audio is considered silence
        if is_silence(reduced_data):
            os.remove(temp_file_path)
            return True
        else:
            # Optionally, save the reduced audio to file for further processing
            wavfile.write(temp_file_path, samplerate, reduced_data.astype(np.int16))
            return False

    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return False


def transcribe_audio(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription


def main():
    model_size = DEFAULT_MODEL_SIZE + ".en"
    model = WhisperModel(model_size, device="cpu", compute_type="int8", num_workers=10)
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    customer_input_transcription = ""

    # Capture a noise profile from the first few seconds of the stream (for example, 2 seconds)
    print("Capturing noise profile...")
    noise_profile = None
    try:
        noise_profile_frames = []
        for _ in range(0, int(16000 / 1024 * 2)):  # 2 seconds of audio
            data = stream.read(1024)
            noise_profile_frames.append(data)

        temp_noise_file = 'temp_noise_profile.wav'
        with wave.open(temp_noise_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(noise_profile_frames))

        samplerate, noise_data = wavfile.read(temp_noise_file)
        noise_profile = noise_data  # Use this as the noise profile
        os.remove(temp_noise_file)

    except Exception as e:
        print(f"Error capturing noise profile: {e}")
    
    try:
        while True:
            chunk_file = "temp_audio_chunk.wav"
            
            # Record audio chunk
            print("_")
            if not record_audio_chunk(audio, stream, noise_profile=noise_profile):
                # Transcribe audio
                transcription = transcribe_audio(model, chunk_file)
                os.remove(chunk_file)
                print("Customer:{}".format(transcription))
                
                # Add customer input to transcript
                customer_input_transcription += "Customer: " + transcription + "\n"
                print('transcribed text: ', transcription)
                
                if transcription != "":
                
                    # Process customer input and get response from AI assistant
                    output = get_rag_answer(transcription)
                    if output:
                        output = output.lstrip()
                        print("AI Assistant:{}".format(output))
                        vs.play_text_to_speech(output)
                    
    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()
