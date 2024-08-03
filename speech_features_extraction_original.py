import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import random as rand
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import History

def extract_features(file_path):
	# Load audio file, refer to https://librosa.org/doc/main/generated/librosa.resample.html for res_type
	audio, sample_rate = librosa.load(file_path, res_type='soxr_vhq')
 
 	# Chromas
	stft = np.abs(librosa.stft(audio))
	chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

	cqt = np.abs(librosa.cqt(y=audio, sr=sample_rate))
	# chroma_cqt = np.mean(librosa.feature.chroma_cqt(C=cqt, sr=sample_rate).T, axis=0)

	# vqt = np.abs(librosa.vqt(audio, sr=sample_rate))
	# chroma_vqt = np.mean(librosa.feature.chroma_vqt(V=vqt, sr=sample_rate).T, axis=0)

	chroma_cens = np.mean(librosa.feature.chroma_cens(C=cqt, sr=sample_rate).T, axis=0)

    # Extract Mel Spectrogram
	mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
    
	# Mel-Frequency Cepstral Coefficients
	mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
	# Standardization: The first approach ((mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-6)) is used to standardize the MFCC features, maintaining their temporal structure while normalizing their scales.
	# Averaging: The second approach (np.mean(mfccs.T, axis=0)) simply averages the MFCC features over time, losing the temporal information but providing a single summary vector.
	mfccs_scaled = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-6)
	# mfccs_scaled = np.mean(mfccs.T, axis=0)
	mfccs_mean = np.mean(mfccs_scaled.T, axis=0)

	# root-mean-square
	rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
 
	# spectral centroid
	spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate).T, axis=0)
 
	# polynomial features
	poly_features = np.mean(librosa.feature.poly_features(y=audio, sr=sample_rate).T, axis=0)
 
 	#	tonal centroid features 
	tonnetz = np.mean(librosa.feature.tonnetz(y=audio, sr=sample_rate).T, axis=0)
 
  	# Zero-Crossing Rate
	zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)

	# Stack features
	features = np.hstack([
		chroma_stft,
		# chroma_cens,
		mel_spectrogram,
		mfccs_mean,
		rms,
		# spectral_centroid,
		# poly_features,
		# tonnetz,
		zcr
	])

	# features = np.vstack([
	# 	zcr,
	# 	mfcc,
	# 	mel_spectrogram,
	# 	chroma_stft
	# ])
 
	# stacked_features = librosa.feature.stack_memory(features.T, n_steps=5)
    
	return features

def load_data(dataset_path):
    features = []
    labels = []
    for speaker in os.listdir(dataset_path):
        speaker_path = os.path.join(dataset_path, speaker)
        if os.path.isdir(speaker_path):
            for file_name in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file_name)
                if file_path.endswith('.flac'):
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(speaker)
    return np.array(features), np.array(labels)

def load_data_toCSV(le, dataset_path, csv_path):
	labels = []
	for speaker in os.listdir(dataset_path):
		speaker_path = os.path.join(dataset_path, speaker)
		if os.path.isdir(speaker_path):
			for file_name in os.listdir(speaker_path):
				file_path = os.path.join(speaker_path, file_name)
				if file_path.endswith('.flac'):
					feature = extract_features(file_path)
					feature_df = pd.DataFrame([feature])
     
					if not os.path.isfile(csv_path):
						feature_df.to_csv(csv_path, index=False)
					else:
						feature_df.to_csv(csv_path, mode='a', header=False, index=False)
					labels.append(speaker)
     
	df = pd.read_csv(csv_path)
	y_encoded = le.fit_transform(labels)
	df['label'] = y_encoded
	df.to_csv(csv_path, index=False, mode='w')

def predict_speaker(le, file_path, model):
    feature = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(feature)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

def predict_speaker_percentage(file_path, model):
    feature = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(feature)
    return prediction

def test_model(le, test_model_path):
	model = tf.keras.models.load_model(test_model_path)
	predicted_speaker = predict_speaker(le, audio_file_path, model)
	predicted_speaker_percentage = predict_speaker_percentage(audio_file_path, model) * 100
	print(f"\nPredicted Speaker ({audio_filename}): {predicted_speaker}")
	print(f"\nPredicted Speaker Similarity ({audio_filename}):  {predicted_speaker_percentage[0][0]:.2f}%")
 
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
 
 

	accuracy = 1.0
	if mode == "train" or mode == "test":
		accuracy = 0.0
	iter = 0

	csv_name, csv_extension = os.path.splitext(csv_path)
	if not os.path.exists(csv_path):
		print("\nNo related csv file found, reading database...")
		load_data_toCSV(le, dataset_path, csv_path)
		df = pd.read_csv(csv_path)
	else:
		print("\nRelated csv file found, loading...")
		df = pd.read_csv(csv_path)

	# filter df row to certain label for training if specified
	if train_speaker != -1:
		df = df[df['label'] == train_speaker] #expected shape of (10,183)

	X = df.drop('label', axis=1)
	y = df['label']
	y_encoded = le.fit_transform(y)
    
	if mode == "analyse":
		audio_path = audio_file_path
		audio, sample_rate = librosa.load(audio_path)
		
		audio_name = os.path.basename(audio_path)
		print(f"\n{audio_name} is chosen...")
	
		#Waveform Plot
		plt.figure(figsize=figsz_config)
		plt.plot(audio)
		plt.title('Waveform')
		plt.xlabel('Time')
		plt.ylabel('Amplitude')
		
		#Spectrogram
		plt.figure(figsize=figsz_config)
		D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
		librosa.display.specshow(D, sr=sample_rate, x_axis='time', y_axis='log')
		plt.colorbar(format='%+2.0f dB')
		plt.title('Spectrogram')
		
		#MFCCs
		mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

		plt.figure(figsize=figsz_config)
		librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
		plt.colorbar()
		plt.title('MFCC')
		plt.tight_layout()
		
		#Chroma Features
		chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)

		plt.figure(figsize=figsz_config)
		librosa.display.specshow(chroma, sr=sample_rate, x_axis='time', y_axis='chroma')
		plt.colorbar()
		plt.title('Chroma Features')
		plt.tight_layout()
	
		#Zero-Crossing Rate
		zcr = librosa.feature.zero_crossing_rate(y=audio)
		plt.figure(figsize=figsz_config)
		plt.plot(zcr[0])
		plt.title('Zero-Crossing Rate')
		plt.xlabel('Time')
		plt.ylabel('Rate')
	
		#Class Distribution
		labels = y_encoded
		label_df = pd.DataFrame(labels, columns=['class'])

		plt.figure(figsize=figsz_config)
		sns.countplot(x='class', data=label_df)
		plt.title('Class Distribution')
		plt.xlabel('Class')
		plt.ylabel('Count')
	
		# Frequency Analysis
		fft = np.fft.fft(audio)
		magnitude = np.abs(fft)
		frequency = np.linspace(0, sample_rate, len(magnitude))
		plt.figure(figsize=figsz_config)
		# this math plot only the positive frequencies
		plt.plot(frequency[:len(frequency)//2], magnitude[:len(magnitude)//2], color='blue', linestyle='-', linewidth=1)
		plt.title('Frequency Spectrum')
		plt.xlabel('Frequency (Hz)')
		plt.ylabel('Magnitude')
		plt.grid(True)

		plt.show()
	
	else: 
		while accuracy <= 0.85 and iter < max_iter:
			iter+=1
			random_seed = rand.randint(0, 9999)
			if mode == "train":
				X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=random_seed)
				

				model = Sequential()
    
				# Input shape: The input shape of a Dense layer represents the shape of the input that the layer expects to receive.
				#               In this case, the input shape is X_train.shape[1], which is the number of features in each sample.
    
				# Dense layer: a fully connected layer with an input shape of X_train.shape[1] and 64 neurons
				# Activation function: relu (Rectified Linear Unit), The activation function is used to introduce non-linearity into the model.
				#                      In this case, we use the ReLU activation function, which is computationally efficient and widely used.
    
				# Dropout layer: This layer randomly sets a fraction of input units to 0 at each update during training time, which helps prevent overfitting.
				#                Here, we set the dropout rate to 0.3, meaning 30% of the input units will be randomly set to 0 at each update.
    
				model.add(Input(shape=(X_train.shape[1],))) # When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead. 
				model.add(Dense(64, activation='relu'))
				model.add(Dropout(0.3))
				model.add(Dense(32, activation='relu'))
				model.add(Dropout(0.3))
					
				# Dense layer: This is the output layer of the model. It has the same number of neurons as the number of unique labels in y_encoded.
				#               The activation function is softmax, which is used for multi-class classification problems.
				#               The softmax function transforms the output of each neuron into a probability distribution over all classes.
				# model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))
				# print(f'\nNumber of unique labels: {len(np.unique(y_encoded))}')
    
				# Explanation: Softmax is not suitable for regression problems, 
    			# 				the following will be used instead
				# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
				# Dense layer: This adds a dense layer to the neural network model with 1 neuron and applies the sigmoid activation function 
    			# 				to the output of that neuron.
				# Explanation: Sigmoid is typically used for binary classification problems, where you need to predict a binary outcome (0 or 1). 
    			# 				The sigmoid activation function outputs a probability between 0 and 1, which can be interpreted as the likelihood of 
       			# 				the positive class.
				# model.add(Dense(1, activation='sigmoid'))
				# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		
				# Explanation: For regression problems, a linear activation function (or no activation function) in the output layer.
				model.add(Dense(1))
				model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    
				history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
	
				loss, accuracy = model.evaluate(X_test, y_test)
				print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
			
				new_model_path = model_path + f"speaker_recognition_model_{random_seed}.keras"
				if accuracy > 0.85:
					keras.saving.save_model(model, new_model_path)
					test_model(le, new_model_path)
					print(f"Model saved at: {new_model_path}")
		
				if show_graphs:
					# Plot training & validation loss values
					plt.figure(figsize=figsz_config)
					plt.plot(history.history['loss'], label='Train Loss')
					plt.plot(history.history['val_loss'], label='Validation Loss')
					plt.title('Model Loss')
					plt.xlabel('Epoch')
					plt.ylabel('Loss')
					plt.legend(loc='upper right')
					plt.grid(True)
		
					# Make predictions on the test set
					y_pred = model.predict(X_test)
					y_pred_classes = np.argmax(y_pred, axis=1)

					# Plot the confusion matrix
					cm = confusion_matrix(y_test, y_pred_classes)
					disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
					disp.plot(cmap=plt.cm.Blues)
					plt.title("Confusion Matrix")
					plt.show()
		
			elif os.path.exists(test_model_path) and mode == "test":
				test_model(test_model_path)
			else:
				print("\nModel not found. Please train the model first.")
	print("\nProcess completed.")

if __name__ == "__main__":
    # ===========================================
	#
	# Default settings
	#
	# ===========================================
	mode = "test" # "train" or "analyse" or "test"
	speaker = 6 # speaker identity for testing
	train_speaker = 6 # speaker identity for training, -1 to train all
	max_iter = 1 # iteration (usually for training model multiple times with random seed)
	show_graphs = False
 
	database_name = "Full Mendeley Data\differentPhrase"
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