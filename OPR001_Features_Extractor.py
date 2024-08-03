
import os
import time
import librosa
import numpy as np
import pandas as pd


class FeatureExtractor:
	def __init__(self, dataset_path, excel_path):
		self.excel_path = excel_path
		self.dataset_path = dataset_path
		# self.file_path = audio_file_path
        
	#region feature extraction functions
	def resample_audio(self, y, sr, target_sr, res_type):
		start_time = time.time()
		y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type=res_type)
		end_time = time.time()
		duration = end_time-start_time
		return y_resampled, duration

	def mean_squared_error(self, y1, y2):
		return np.mean((y1-y2)**2)

	def signal_to_noise_ratio(self, y_original, y_resampled):
		signal_power = np.mean(y_original**2)
		noise_power = np.mean((y_original-y_resampled)**2)
		if noise_power == 0:
			return 'No noise'  # Infinite SNR if there's no noise
		snr = 10 * np.log10(signal_power/noise_power)
		return snr

	def noise(self, data):
		'''
		- Robustness
		- Generalization
		- Data Augmentation
		'''
		noise_amp = 0.035*np.random.uniform()*np.amax(data)
		data = data + noise_amp*np.random.normal(size=data.shape[0])
		return data

	def stretch(self, data, rate=0.8):
		'''
		- Data Augmentation
		- Speaker Variability
		- Noise Reduction
		- Synchronization
		'''
		return librosa.effects.time_stretch(data, rate=rate)

	def shift(self, data):
		'''
		- Temporal Alignment
		- Noise Reduction
		- Accent and Dialect Adaptation
		- Error Correction
		'''
		shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
		return np.roll(data, shift_range)

	def pitch(self, data, sampling_rate=44100, pitch_factor=0.7):
		'''
		- Frequency Alteration
		- Formant Preservation
		- Speech Intelligibility
		- Acoustic Model Limitations
		'''
		return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

	def extract_features(self, data, sample_rate=44100):
		'''
		Zero-Crossing Rate (ZCR):
		This is a measure of how many times the audio signal crosses the zero amplitude line per second.
		It's often used to analyze the noisiness or the percussiveness of an audio signal.
		A high ZCR typically indicates a signal with a lot of sudden changes in amplitude, like percussive sounds.
		'''
		zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)

		'''
		Chroma Short-Time Fourier Transform (Chroma_stft):
		This feature captures the energy distribution of audio signal frequencies into 12 different pitch classes.
		It is particularly useful for analyzing harmonic and melodic content in music.
		Chroma features are based on the pitch class (e.g., C, C#, D, etc.) rather than the exact frequencies, which makes it robust to changes in pitch.
		'''
		stft = np.abs(librosa.stft(data))
		chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

		'''
		Mel-Frequency Cepstral Coefficients (MFCC):
		MFCCs are coefficients that represent the short-term power spectrum of a sound.
		They are derived from the cepstrum of an audio signal and are used to describe the shape of the power spectrum.
		MFCCs are widely used in speech and audio processing because they provide a compact representation of the spectral properties of the sound.
		'''
		mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)

		'''
		Root Mean Square Value (RMS):
		This is a measure of the average power of an audio signal.
		RMS value quantifies the magnitude of the varying audio signal and is often used to determine the loudness of the sound.
		It provides a measure of the signal's energy.
		'''
		rms = np.mean(librosa.feature.rms(y=data).T, axis=0)

		'''
		Mel Spectrogram:
		This is a visual representation of the spectrum of frequencies in a sound signal as it varies with time,
		using the Mel scale for frequency representation.
		The Mel scale is a perceptual scale of pitches which approximates the human ear's response to different frequencies.
		Mel spectrograms are useful for visualizing and analyzing audio data, particularly for tasks like speech recognition and music genre classification.
		'''
		mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)

		result = np.array([])
		result = np.hstack((result, zcr, chroma_stft, mfcc, rms, mel))

		return result

	def get_features(self, file_path):
		# duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
		data, sample_rate = librosa.load(file_path, sr=None, duration=2.5, offset=0.6)
		# resampling
		data = librosa.resample(data, orig_sr=sample_rate, target_sr=44100, res_type='soxr_vhq')

		# without augmentation
		audio1 = self.extract_features(data)
		result = np.array(audio1)

		# data with noise
		data_noise = self.noise(data)
		audio2 = self.extract_features(data_noise)
		result = np.vstack((result, audio2))

		# data with stretching and pitching
		data_stretch = self.stretch(data)
		data_stretch_pitch = self.pitch(data_stretch, sample_rate)
		audio3 = self.extract_features(data_stretch_pitch)
		result = np.vstack((result, audio3))

		return result
	#endregion feature extraction functions
