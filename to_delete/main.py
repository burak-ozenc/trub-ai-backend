# import os
# import librosa
# import uuid
# # from fastapi import FastAPI, File, UploadFile
# import warnings
# from pydub import AudioSegment
# import matplotlib.pyplot as plt
# import librosa.display
# import numpy as np
# from scipy.signal import butter, lfilter, filtfilt
# import noisereduce as nr
# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse
# import json
# import ollama
# from pydantic import BaseModel
# # import debugpy
# 
# 
# # # For Debugger 
# # debugpy.listen(8000)
# # debugpy.wait_for_client()
# 
# # Initialize FastAPI app
# app = FastAPI()
# 
# # Create a folder for storing uploaded files (optional)
# UPLOAD_DIR = "C:\\recordings"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# AudioSegment.converter = "C:\\ffmpeg\\bin\\ffplay.exe"
# 
# 
# # @app.post("/process-audio")
# # async def process_audio(audioData: UploadFile = File(...)):
# #     try:
# #         # Check for 'blob' or empty filename, and generate a unique filename
# #         file_name = audioData.filename
# #         if not file_name or file_name == "blob":
# #             file_name = f"audio_{uuid.uuid4().hex}.wav"
# # 
# #         file_path = os.path.join(UPLOAD_DIR, file_name)
# # 
# #         # Save the uploaded file
# #         with open(file_path, "wb") as f:
# #             content = await audioData.read()
# #             f.write(content)
# # 
# #         # Log the saved file path
# #         print(f"File saved at: {file_path}")
# # 
# #         # Placeholder for processing logic
# #         feedback = analyze_audio(file_path)
# # 
# #         return {"feedback": feedback, "file_path": file_path}
# #     except Exception as e:
# #         print(f"Error: {str(e)}")  # Log the error for debugging
# #         return JSONResponse(status_code=500, content={"message": str(e)})
# 
# # @app.post("/process-audio")
# # async def process_audio(audioData: UploadFile = File(...), guidance: str = Form(...)):
# #     try:
# #         # Log the guidance text
# #         print(f"Received guidance: {guidance}")
# # 
# #         file_name = audioData.filename
# #         if not file_name or file_name == "blob":
# #             file_name = f"audio_{uuid.uuid4().hex}.wav"
# # 
# #         file_path = os.path.join(UPLOAD_DIR, file_name)
# # 
# #         with open(file_path, "wb") as f:
# #             content = await audioData.read()
# #             f.write(content)
# # 
# #         feedback = analyze_audio(file_path)
# #         return {"feedback": feedback, "file_path": file_path}
# #     except Exception as e:
# #         print(f"Error: {str(e)}")
# #         return JSONResponse(status_code=500, content={"message": str(e)})
# 
# 
# def analyze_audio_for_llm(file_path: str) -> dict:
#     """Convert audio analysis into structured data for LLM"""
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         y, sr = librosa.load(file_path, sr=None)
#         y_filtered = apply_bandpass_filter(y, sr)
#         y_denoised = remove_background_noise(y_filtered, sr)
#         y_enhanced = enhance_trumpet_signal(y_denoised)
# 
#         return {
#             "tone_quality": analyze_tone_quality(y_enhanced, sr),
#             "flexibility": analyze_flexibility(y_enhanced, sr),
#             "rhythm_timing": analyze_rhythm_timing(y_enhanced, sr),
#             "expression": analyze_expression(y_enhanced, sr)
#         }
# 
# 
# def create_llm_prompt(analysis_results: dict, guidance: str) -> str:
#     return f"""As a music teacher, provide feedback based on this trumpet performance analysis and student's question.
# 
# Analysis Results:
# {json.dumps(analysis_results, indent=2)}
# 
# Student's Question/Guidance:
# {guidance}
# 
# Provide specific, constructive feedback addressing both the technical aspects and the student's question."""
# 
# 
# async def get_llm_feedback(prompt: str) -> str:
#     try:
#         print("sending a prompt")
#         response = ollama.generate(
#             model="deepseek-r1:7b",
#             prompt= prompt
#         )
#         print("answer acquired")
#         test = response["response"]
#         print(test)
#         return test
#     except Exception as e:
#         print(f"LLM Error: {str(e)}")
#         return "Error getting LLM feedback"
# 
# 
# @app.post("/process-audio")
# async def process_audio(audioData: UploadFile = File(...), guidance: str = Form(...)):
#     try:
#         print("method starting")
#         file_name = audioData.filename or f"audio_{uuid.uuid4().hex}.wav"
#         file_path = os.path.join(UPLOAD_DIR, file_name)
# 
#         with open(file_path, "wb") as f:
#             content = await audioData.read()
#             f.write(content)
# 
#         analysis_results = analyze_audio_for_llm(file_path)
#         print(analysis_results)
#         prompt = create_llm_prompt(analysis_results, guidance)
#         llm_feedback = await get_llm_feedback(prompt)
# 
#         return {
#             "feedback": llm_feedback,
#             "file_path": file_path,
#             "analysis": analysis_results
#         }
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return JSONResponse(status_code=500, content={"message": str(e)})
# 
# 
# def analyze_audio(file_path: str) -> str:
#     try:
#         # Suppress all warnings within this block
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
# 
#             y, sr = librosa.load(file_path, sr=None)
#             # Step 2: Preprocessing
#             # Apply band-pass filter
#             y_filtered = apply_bandpass_filter(y, sr)
#             # Remove background noise
#             y_denoised = remove_background_noise(y_filtered, sr)
#             # Enhance trumpet signal
#             y_enhanced = enhance_trumpet_signal(y_denoised)
# 
#             tone_quality_feedback = analyze_tone_quality(y_enhanced, sr)
#             flexibility_feedback = analyze_flexibility(y_enhanced, sr)
#             rhythm_timing_feedback = analyze_rhythm_timing(y_enhanced, sr)
#             expression_feedback = analyze_expression(y_enhanced, sr)
# 
#             # Combine feedback into a single response
#             feedback = "\n".join([
#                 tone_quality_feedback
#                 ,
#                 flexibility_feedback,
#                 rhythm_timing_feedback,
#                 expression_feedback
#             ])
# 
#             print(feedback)
#     except Exception as e:
#         print(f"Error loading audio: {e}")
#         return f"Error processing the audio file: {e}"
# 
# 
# def detect_pitch(file_path):
#     y, sr = librosa.load(file_path, sr=None)
#     pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
# 
#     # Get the most prominent pitch
#     pitch_values = []
#     for frame in range(pitches.shape[1]):
#         index = magnitudes[:, frame].argmax()
#         pitch = pitches[index, frame]
#         if pitch > 0:  # Filter out unvoiced frames
#             pitch_values.append(pitch)
# 
#     return pitch_values
# 
# 
# def analyze_intonation(detected_pitch, target_pitch):
#     deviation = detected_pitch - target_pitch
#     if abs(deviation) < 1:  # Tolerance for near-perfect intonation
#         return "Perfect intonation!"
#     elif deviation > 0:
#         return f"You're sharp by {deviation:.2f} Hz."
#     else:
#         return f"You're flat by {abs(deviation):.2f} Hz."
# 
# 
# def analyze_timing(file_path):
#     y, sr = librosa.load(file_path, sr=None)
#     tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
#     return tempo, beats
# 
# 
# def visualize_spectrogram(file_path):
#     y, sr = librosa.load(file_path, sr=None)
#     S = librosa.feature.melspectrogram(y=y, sr=sr)
#     S_dB = librosa.power_to_db(S, ref=np.max)
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Mel-frequency spectrogram')
#     plt.tight_layout()
#     plt.show()
# 
# 
# def band_pass_filter(data, sr, lowcut=233.0, highcut=2118.90):
#     nyquist = 0.5 * sr
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = butter(N=4, Wn=[low, high], btype='band')
#     y = lfilter(b, a, data)
#     return y
# 
# 
# def process_raw_audio(file_path):
#     y, sr = librosa.load(file_path, sr=None)  # Load raw audio
#     filtered_audio = band_pass_filter(y, sr)
#     return filtered_audio, sr
# 
# 
# def reduce_noise(audio, sr):
#     # Estimate noise profile from the first 0.5 seconds
#     noise_profile = audio[:int(sr * 0.5)]
#     reduced_audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_profile)
#     return reduced_audio
# 
# 
# def apply_bandpass_filter(y, sr, low_cutoff=233, high_cutoff=2118.90):
#     # Design a Butterworth band-pass filter
#     nyquist = 0.5 * sr
#     low = low_cutoff / nyquist
#     high = high_cutoff / nyquist
#     b, a = butter(N=4, Wn=[low, high], btype='band')
#     filtered_signal = filtfilt(b, a, y)
#     return filtered_signal
# 
# 
# def remove_background_noise(y, sr):
#     # Create a noise profile using the first 0.5 seconds (assumed to be noise)
#     noise_sample = y[:int(0.5 * sr)]
#     reduced_noise_signal = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample)
#     return reduced_noise_signal
# 
# 
# def enhance_trumpet_signal(y):
#     # Apply dynamic range compression
#     compressed_signal = librosa.effects.percussive(y, margin=1.0)
#     return compressed_signal
# 
# 
# def analyze_tone_quality(y, sr):
#     harmonic, percussive = librosa.effects.hpss(y)
#     harmonic_ratio = np.mean(harmonic) / (np.mean(harmonic) + np.mean(percussive))
# 
#     if harmonic_ratio > 0.7:
#         return "Tone Quality: Excellent harmonic richness and clarity."
#     elif harmonic_ratio > 0.5:
#         return "Tone Quality: Good, but could use more harmonic clarity."
#     else:
#         return "Tone Quality: Needs improvement. Focus on a richer and more resonant tone."
# 
# 
# def analyze_flexibility(y, sr):
#     pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
#     pitch_changes = np.diff(np.max(pitches, axis=0))
# 
#     # Measure average pitch change and variability
#     avg_pitch_change = np.mean(np.abs(pitch_changes))
#     if avg_pitch_change > 10:
#         return "Flexibility: Excellent smoothness in note transitions."
#     elif avg_pitch_change > 5:
#         return "Flexibility: Good, but practice smoother transitions between notes."
#     else:
#         return "Flexibility: Needs improvement. Work on smoother note changes."
# 
# 
# def analyze_rhythm_timing(y, sr):
#     onset_env = librosa.onset.onset_strength(y=y, sr=sr)
#     tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
# 
#     if 40 < tempo < 180:
#         return f"Rhythm and Timing: Good tempo at {"{:.5f}".format(float(tempo))} BPM. Keep it steady!"
#     else:
#         return (f"Rhythm and Timing: Irregular tempo detected ({"{:.5f}".format(float(tempo))} BPM). Practice with a "
#                 f"metronome.")
# 
# 
# def analyze_expression(y, sr):
#     S, phase = librosa.magphase(librosa.stft(y))
#     rms = librosa.feature.rms(S=S)
# 
#     avg_dynamic_range = np.max(rms) - np.min(rms)
#     if avg_dynamic_range > 0.05:
#         return "Expression: Great dynamic range and expressiveness."
#     elif avg_dynamic_range > 0.02:
#         return "Expression: Good, but try incorporating more dynamic variation."
#     else:
#         return "Expression: Needs more variation in dynamics. Focus on musical phrasing."
# 
# 
# class GenerateRequest(BaseModel):
#     model: str
#     prompt: str