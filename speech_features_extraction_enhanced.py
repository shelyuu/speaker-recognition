import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import random as rand
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import History

#region feature extraction functions
def resample_audio(y, sr, target_sr, res_type):
    start_time = time.time()
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type=res_type)
    end_time = time.time()
    duration = end_time-start_time
    return y_resampled, duration

def mean_squared_error(y1, y2):
    return np.mean((y1-y2)**2)

def signal_to_noise_ratio(y_original, y_resampled):
    signal_power = np.mean(y_original**2)
    noise_power = np.mean((y_original-y_resampled)**2)
    if noise_power == 0:
        return 'No noise'  # Infinite SNR if there's no noise
    snr = 10 * np.log10(signal_power/noise_power)
    return snr

def noise(data):
    '''
    - Robustness
    - Generalization
    - Data Augmentation
    '''
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    '''
    - Data Augmentation
    - Speaker Variability
    - Noise Reduction
    - Synchronization
    '''
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    '''
    - Temporal Alignment
    - Noise Reduction
    - Accent and Dialect Adaptation
    - Error Correction
    '''
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate=44100, pitch_factor=0.7):
    '''
    - Frequency Alteration
    - Formant Preservation
    - Speech Intelligibility
    - Acoustic Model Limitations
    '''
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features(data, sample_rate=44100):
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

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, sr=None, duration=2.5, offset=0.6)

    # resampling
    data = librosa.resample(data, orig_sr=sample_rate, target_sr=44100, res_type='soxr_vhq')

    # without augmentation
    audio1 = extract_features(data)
    result = np.array(audio1)

    # data with noise
    data_noise = noise(data)
    audio2 = extract_features(data_noise)
    result = np.vstack((result, audio2))

    # data with stretching and pitching
    data_stretch = stretch(data)
    data_stretch_pitch = pitch(data_stretch, sample_rate)
    audio3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, audio3))

    return result
#endregion feature extraction functions

def load_data(dataset_path):
	features, labels = [], []
	for speaker in os.listdir(dataset_path):
		speaker_path = os.path.join(dataset_path, speaker)
		if os.path.isdir(speaker_path):
			for file_name in os.listdir(speaker_path):
				file_path = os.path.join(speaker_path, file_name)
				if file_path.endswith('.flac') or file_path.endswith('.wav') or file_path.endswith('.mp3'):
					feature = get_features(file_path)
					for type in feature:
						features.append(type)
						labels.append(speaker)
	return features, labels

def save_data_toXLSX(dataset_path, excel_path):
	print("Extracting features and saving to Excel...")
	X, Y = load_data(dataset_path)
	df = pd.DataFrame(X)
	df['SPEAKER'] = Y
	df.to_excel(f"{excel_path}", index=False)

def load_data_toCSV(le, dataset_path, csv_path):
	labels = []
	for speaker in os.listdir(dataset_path):
		speaker_path = os.path.join(dataset_path, speaker)
		if os.path.isdir(speaker_path):
			for file_name in os.listdir(speaker_path):
				file_path = os.path.join(speaker_path, file_name)
				if file_path.endswith('.flac') or file_path.endswith('.wav') or file_path.endswith('.mp3'):
					feature = get_features(file_path)
					feature_df = pd.DataFrame([feature])
     
					if not os.path.isfile(csv_path):
						feature_df.to_csv(csv_path, index=False)
					else:
						feature_df.to_csv(csv_path, mode='a', header=False, index=False)
					labels.append(speaker)
     
	df = pd.read_csv(csv_path)
	# y_encoded = le.fit_transform(labels)
	df['SPEAKER'] = labels
	df.to_csv(csv_path, index=False, mode='w')
 
def shorten_name(input_string):
    name_parts = input_string.split('\\')
    d_name = name_parts[0]
    d_initials = ''.join(word[0].upper() for word in d_name.split())

    after_backslash = name_parts[1:]
    after_backslash_initials = ''.join(word[0].upper() for word in after_backslash)

    short_name = f"{d_initials}_{after_backslash_initials}"

    return short_name

def get_a_random_file(folder_path):
    # Randomly select a file
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path {folder_path} is not a directory.")
    
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    if not files:
        raise ValueError(f"The directory {folder_path} does not contain any files.")
    
    random_file = rand.choice(files)
    
    return random_file

def load_save_excel(dataset_path, excel_path):
	fea_name, extension = os.path.splitext(excel_path)
	if not os.path.exists(excel_path):
		print("\nNo related features file found, reading database...")
		save_data_toXLSX(dataset_path, excel_path)
		# load_data_toCSV(le, dataset_path, csv_path)
		# df = pd.read_csv(csv_path)
	else:
		print("\nRelated features file found, loading...")
		# df = pd.read_csv(csv_path)
	df = pd.read_excel(excel_path)
	return df

def main(arg1, arg2, arg3, arg4, arg5, arg6):
	print(f"\nArguments received: {arg1}, {arg2}, {arg3}, {arg4}, {arg5}, {arg6}")
	global audio_file_path
	global audio_filename
	# audio_file_path = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("flac files","*.flac"),("mp3 files","*.mp3"),("wav files","*.wav"),("all files","*.*")))
	
	# ===========================================
	#
	# settings
	#
	# ===========================================
	mode = arg1 # "train" or "analyse" or "test"
	speaker = arg2 # speaker identity for testing
	train_speaker = arg3 # speaker identity for training, -1 to ignore
	max_iter = arg4 # iteration (usually for training model with random seed)
	show_graphs = arg5
	dataset_path = arg6
	# end settings	=====================

	figsz_config = (8, 3)
	le = LabelEncoder()
 
	last_backslash_index = dataset_path.rfind('\\')
	second_last_backslash_index = dataset_path.rfind('\\', 0, last_backslash_index)
	database_name = dataset_path[second_last_backslash_index + 1:]
 
	audio_filename = get_a_random_file(f'{dataset_path}\\{speaker}\\') # f"{speaker}-{rand.randint(1, 10)}.flac"
	audio_file_path = f"{dataset_path}\\{speaker}\\{audio_filename}"
	model_path = "model\\"
	test_model_path = "model\speaker_recognition_model.keras"
	csv_path = f'data\\features_labels_{shorten_name(database_name)}.csv'
	xlsx_path = f'data\\features_labels_{shorten_name(database_name)}.xlsx'
 
 

	accuracy = 1.0
	if mode == "train" or mode == "test":
		accuracy = 0.0
	iter = 0
 
	print("\nStart loading features.")
	df = load_save_excel(dataset_path, xlsx_path)
	print("\nEnd loading features.")
	# filter df row to certain label for training if specified
	# if train_speaker != -1:
	# 	df = df[df['label'] == train_speaker] #expected shape of (10,183)

	# X = df.drop('label', axis=1)
	# y = df['label']
	# y_encoded = le.fit_transform(y)
    
	# while accuracy <= 0.85 and iter < max_iter:
	# 	iter+=1
	# 	random_seed = rand.randint(0, 9999)
	# 	if mode == "train":
	# 		X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=random_seed)
			

	# 		model = Sequential()

	# 		# Input shape: The input shape of a Dense layer represents the shape of the input that the layer expects to receive.
	# 		#               In this case, the input shape is X_train.shape[1], which is the number of features in each sample.

	# 		# Dense layer: a fully connected layer with an input shape of X_train.shape[1] and 64 neurons
	# 		# Activation function: relu (Rectified Linear Unit), The activation function is used to introduce non-linearity into the model.
	# 		#                      In this case, we use the ReLU activation function, which is computationally efficient and widely used.

	# 		# Dropout layer: This layer randomly sets a fraction of input units to 0 at each update during training time, which helps prevent overfitting.
	# 		#                Here, we set the dropout rate to 0.3, meaning 30% of the input units will be randomly set to 0 at each update.

	# 		model.add(Input(shape=(X_train.shape[1],))) # When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead. 
	# 		model.add(Dense(64, activation='relu'))
	# 		model.add(Dropout(0.3))
	# 		model.add(Dense(32, activation='relu'))
	# 		model.add(Dropout(0.3))
				
	# 		# Dense layer: This is the output layer of the model. It has the same number of neurons as the number of unique labels in y_encoded.
	# 		#               The activation function is softmax, which is used for multi-class classification problems.
	# 		#               The softmax function transforms the output of each neuron into a probability distribution over all classes.
	# 		# model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))
	# 		# print(f'\nNumber of unique labels: {len(np.unique(y_encoded))}')

	# 		# Explanation: Softmax is not suitable for regression problems, 
	# 		# 				the following will be used instead
	# 		# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# 		# Dense layer: This adds a dense layer to the neural network model with 1 neuron and applies the sigmoid activation function 
	# 		# 				to the output of that neuron.
	# 		# Explanation: Sigmoid is typically used for binary classification problems, where you need to predict a binary outcome (0 or 1). 
	# 		# 				The sigmoid activation function outputs a probability between 0 and 1, which can be interpreted as the likelihood of 
	# 		# 				the positive class.
	# 		# model.add(Dense(1, activation='sigmoid'))
	# 		# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	# 		# Explanation: For regression problems, a linear activation function (or no activation function) in the output layer.
	# 		model.add(Dense(1))
	# 		model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	# 		history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

	# 		loss, accuracy = model.evaluate(X_test, y_test)
	# 		print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
		
	# 		new_model_path = model_path + f"speaker_recognition_model_{random_seed}.keras"
	# 		if accuracy > 0.85:
	# 			keras.saving.save_model(model, new_model_path)
	# 			test_model(le, new_model_path)
	# 			print(f"Model saved at: {new_model_path}")
	
	# 		if show_graphs:
	# 			# Plot training & validation loss values
	# 			plt.figure(figsize=figsz_config)
	# 			plt.plot(history.history['loss'], label='Train Loss')
	# 			plt.plot(history.history['val_loss'], label='Validation Loss')
	# 			plt.title('Model Loss')
	# 			plt.xlabel('Epoch')
	# 			plt.ylabel('Loss')
	# 			plt.legend(loc='upper right')
	# 			plt.grid(True)
	
	# 			# Make predictions on the test set
	# 			y_pred = model.predict(X_test)
	# 			y_pred_classes = np.argmax(y_pred, axis=1)

	# 			# Plot the confusion matrix
	# 			cm = confusion_matrix(y_test, y_pred_classes)
	# 			disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
	# 			disp.plot(cmap=plt.cm.Blues)
	# 			plt.title("Confusion Matrix")
	# 			plt.show()
	
	# 	elif os.path.exists(test_model_path) and mode == "test":
	# 		test_model(test_model_path)
	# 	else:
	# 		print("\nModel not found. Please train the model first.")
	print("\nProcess completed.")

if __name__ == "__main__":
    # ===========================================
	#
	# Default settings
	#
	# ===========================================
	mode = "extract" # "extract" or "train" or "test"
	speaker = 6 # speaker identity for testing
	train_speaker = 6 # speaker identity for training, -1 to train all
	max_iter = 1 # iteration (usually for training model multiple times with random seed)
	show_graphs = False
 
	database_name = "Mendeley Data\differentPhrase"
	dataset_path = f"audio\speakers\{database_name}"
	# end settings	=====================
 
 
	# Set default values
	default_arg1 = mode
	default_arg2 = speaker
	default_arg3 = train_speaker
	default_arg4 = max_iter
	default_arg5 = show_graphs
	default_arg6 = dataset_path

	# print(len(sys.argv))
	# Check if arguments are provided
	if len(sys.argv) == 7:
		arg1 = sys.argv[1]
		arg2 = sys.argv[2]
		arg3 = sys.argv[3]
		arg4 = sys.argv[4]
		arg5 = sys.argv[5]
		arg6 = sys.argv[6]
	else:
		arg1 = default_arg1
		arg2 = default_arg2
		arg3 = default_arg3
		arg4 = default_arg4
		arg5 = default_arg5
		arg6 = default_arg6

	main(arg1, arg2, arg3, arg4, arg5, arg6)