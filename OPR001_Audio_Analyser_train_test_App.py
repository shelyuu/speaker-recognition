#region imports
import io
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import time
import soxr
import random as rand
import itertools
import numpy as np
import librosa
import librosa.display
import threading
import subprocess
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('agg')  # Use the 'agg' backend for Matplotlib
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
from tensorflow.keras.models import load_model
import pandas as pd
import pickle
import pygame
import re

from OPR001_Features_Extractor import FeatureExtractor
#endregion

def run_subclass():
    arg1 = dropdown_mode.get() # mode = "train" # "train" or "analyse" or "test"
    arg2 = -1 if dropdown_speaker_var2.get() == "**ALL**" else dropdown_speaker_var2.get() # speaker = 2 # speaker identity for testing
    arg3 = -1 if dropdown_speaker_var2.get() == "**ALL**" else dropdown_speaker_var2.get() # train_speaker = 1 # speaker identity for training, -1 to ignore
    arg4 = 1 #str(scale_maxiter_var.get()) # max_iter = 1 # iteration (usually for training model with random seed)
    arg5 = False #str(bool(var1_show_graphs.get())) # show_graphs = False
    arg6 = database_entry.get() # dataset_path = f"audio\speakers\{database_name}"
    
    mode = arg1 # "train" or "analyse" or "test"
    speaker = arg2 # speaker identity for testing
    train_speaker = arg3 # speaker identity for training, -1 to ignore
    max_iter = arg4 # iteration (usually for training model with random seed)
    show_graphs = arg5
    dataset_path = arg6
    # end settings	=====================

    figsz_config = (8, 3)
    # le = LabelEncoder()

    last_backslash_index = dataset_path.rfind('\\')
    second_last_backslash_index = dataset_path.rfind('\\', 0, last_backslash_index)
    database_name = dataset_path[second_last_backslash_index + 1:]

    # audio_filename = get_a_random_file(f'{dataset_path}\\{speaker}\\') # f"{speaker}-{rand.randint(1, 10)}.flac"
    # audio_file_path = f"{dataset_path}\\{speaker}\\{audio_filename}"
    model_path = "model\\"
    test_model_path = "model\speaker_recognition_model.keras"
    csv_path = f'data\\features_labels_{shorten_name(database_name)}.csv'
    xlsx_path = f'data\\features_labels_{shorten_name(database_name)}.xlsx'
 
    fea_name, extension = os.path.splitext(xlsx_path)
    if not os.path.exists(xlsx_path):
        log("\nNo related features file found, reading database...")
        extractor = FeatureExtractor(dataset_path, xlsx_path)
        save_data_toXLSX(extractor)
    else:
        log("\nRelated features file found, no features file is created. ")
    df = pd.read_excel(xlsx_path)
    
    return df

def load_data(extractor):
    features, labels = [], []
    for speaker in os.listdir(extractor.dataset_path):
        speaker_path = os.path.join(extractor.dataset_path, speaker)
        if os.path.isdir(speaker_path):
            for file_name in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file_name)
                if file_path.endswith('.flac') or file_path.endswith('.wav') or file_path.endswith('.mp3'):
                    feature = extractor.get_features(file_path)
                    for type in feature:
                        features.append(type)
                        labels.append(speaker)
    log(f"Extracted {len(features)} features and labels")
    return features, labels

def save_data_toXLSX(extractor):
    log(f"Extracting features and saving to Excel...: {extractor.excel_path}")
    X, Y = load_data(extractor)
    df = pd.DataFrame(X)
    df['SPEAKER'] = Y
    df.to_excel(extractor.excel_path, index=False)

#region Submit buttons
def submit_tab1():
    selected_folder = folder_path.get()
    selected_plot_type = dropdown_var1.get()
    # selected_option2 = dropdown_var2.get()
    
    if not selected_folder:
        messagebox.showwarning("Warning", "Please select a folder")
        log("Warning: No folder selected")
    else:
        result = f"Folder: {selected_folder}\nPlot Type: {selected_plot_type}"
        messagebox.showinfo("Graph Plotting Start", result)
        log(f"Audio Visuallizing Details:\n{result}")
        
        loading_label_tab1.config(text="Loading, please wait...")
        threading.Thread(target=tab1_process_audio, args=(selected_folder, selected_plot_type)).start()

def submit_tab2():
    try :
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        selected_folder = database_path.get()
        if not selected_folder:
            messagebox.showwarning("Warning", "Please select a Database folder")
            log("Warning: No Database folder selected")
            return
        
        loading_label_tab2.config(text="Loading, please wait... may take minutes to process")
        tab2.update()
        
        df = run_subclass()
        printed_messages = sys.stdout.getvalue()
        log(printed_messages)
        sys.stdout = old_stdout
        
        if df is None:
            log("Features extraction failed")
            return
        else:
            log("Features extraction completed successfully")
            loading_label_tab2.config(text="Processing completed")
    except Exception as e:
        log(f"Exception: {e}")
    finally:
        tab2.update()

def submit_tab3():
    selected_audio = audio_path.get()
    selected_model = model_path.get()
    selected_features = features_path.get()
    
    if not selected_audio:
        messagebox.showwarning("Warning", "Please select an Audio file (.wav .mp3 .flac)")
        log("Warning: No Audio selected")
    elif not selected_model:
        messagebox.showwarning("Warning", "Please select an Model (.h5)")
        log("Warning: No Model selected")
    elif not selected_features:
        messagebox.showwarning("Warning", "Please select an Features File (.xlsx)")
        log("Warning: No Features selected")
    else:
        result = f"Audio File: {selected_audio}\nModel: {selected_model}\nFeatures File: {selected_features}"
        messagebox.showinfo("Audio Analysis Start", result)
        log(f"Audio Analysis Details:\n{result}")
        loading_label_tab3.config(text="Loading, please wait...")
        threading.Thread(target=tab3_analysis, args=(selected_audio, selected_model, selected_features)).start()
#endregion 

#region Utility Functions

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

def loading_indicator(stop_event):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if stop_event.is_set():
            break
        log(f'Loading... {c}')
        time.sleep(0.1)
    log('Loading... Done!')

# Function to browse folder
def browse_folder():
    folder_selected = filedialog.askdirectory()
    
    if folder_path and selected_tab == 0:
        folder_path.set(folder_selected)
    elif database_path and selected_tab == 1:
        database_path.set(folder_selected)
        update_dropdown(database_path.get())
 
def browse_audio():
    audio_selected = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac")])
    if audio_selected: 
        audio_path.set(audio_selected)
        tab3_draw_audio(audio_path.get())
        
def browse_model():
    model_selected = filedialog.askopenfilename(filetypes=[("Model Files", "*.h5")])
    if model_selected: 
        model_path.set(model_selected)
        
def browse_features():
    features_selected = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if features_selected: 
        features_path.set(features_selected)

# Update dropdown for Tab 2 Speaker according to folder selected
def update_dropdown(folder_path):
    print(folder_path)
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    dropdown_speaker_var2.set(folders[0] if folders else "No folders found")
    menu = dropdown_speaker_menu2['menu']
    menu.delete(0, 'end')
    menu.add_command(label="**ALL**", command=tk._setit(dropdown_speaker_var2, "**ALL**"))
    for folder in folders:
        menu.add_command(label=folder, command=tk._setit(dropdown_speaker_var2, folder))
    
    dropdown_speaker_var2.set("**ALL**")
    # log(f"Folder loaded: {folder_path}\n")

# Update the Scale 
def update_scale_value(val):
    scale_value_label.config(text=f"Max Iteration: {scale_maxiter_var.get()}")

# Set the Tab index
def on_tab_selected(event):
    global selected_tab
    selected_tab = event.widget.index("current")
    # print(f'current tab : {selected_tab}')

# Log messages to the text area
def log(message):
    if selected_tab == 0:
        log_area_tab1.config(state=tk.NORMAL)
        log_area_tab1.insert(tk.END, message + "\n")
        log_area_tab1.config(state=tk.DISABLED)
        log_area_tab1.yview(tk.END)
        
    if selected_tab == 1:
        log_area_tab2.config(state=tk.NORMAL)
        log_area_tab2.insert(tk.END, message + "\n")
        log_area_tab2.config(state=tk.DISABLED)
        log_area_tab2.yview(tk.END)
        
    if selected_tab == 2:
        log_area_tab3.config(state=tk.NORMAL)
        log_area_tab3.insert(tk.END, message + "\n")
        log_area_tab3.config(state=tk.DISABLED)
        log_area_tab3.yview(tk.END)
#endregion 

#region Tab Functions
def tab3_analysis(audio_path, model_path, features_path):
    global plot_canvas_tab3
    
    # Clear existing plot
    for widget in plot_frame_tab3.winfo_children():
        widget.destroy()
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot the waveplot
    audio_name = os.path.basename(audio_path)
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title='Waveplot of ' + audio_name)
    
    # Embed the plot in the Tkinter window
    plot_canvas_tab3 = FigureCanvasTkAgg(fig, master=plot_frame_tab3)
    plot_canvas_tab3.draw()
    plot_canvas_tab3.get_tk_widget().pack(expand=1, fill='both')

    # Get speaker ID and sample
    match_id = re.search(r'(\d+)-\d+\.flac$', audio_path)
    speaker_id = 'Unknown' if not match_id else match_id.group(1)
    match_sample = re.search(r'-(\d+)\.flac$', audio_path)
    speaker_sample = 'Unknown' if not match_sample else match_sample.group(1) 
    
    # Update the label with the file details
    log(f"Speaker ID - Sample: {speaker_id} - {speaker_sample}\nSample Rate: {sr} Hz\nDuration: {duration:.2f} seconds")

    # Prediction Part ----------------------------------------------------------------------------------------------

    # Load the scaler, PCA, OneHotEncoder, model
    with open(r'packages/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open(r'packages/pca.pkl', 'rb') as f:
        pca = pickle.load(f)

    with open(r'packages/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    model = load_model(model_path)

    # Replace 'your_file.xlsx' with the path to your Excel file
    lookup_features = features_path

    # Read the Excel file
    df = pd.read_excel(lookup_features)
    df['SAMPLE'] = df.groupby('SPEAKER').cumcount()

    # Lookup features
    df = df.loc[(df.SPEAKER == int(speaker_id))&(df.SAMPLE == int(speaker_sample))]

    # Transform
    X = df.iloc[:,:-2].values 
    X = scaler.transform(X)
    X = pca.transform(X)
    X = np.expand_dims(X, axis=2)

    # Predict
    probas = model.predict(X)
    top_10_lables_idx = np.argsort(-probas)
    top_10_labels = [lst[:10] for lst in [encoder.categories_[0][i] for i in top_10_lables_idx]]

    # Output
    prediction = ', '.join(map(str, top_10_labels))
    log(f"Top 10 prediction sorted by confidence: {prediction}")
    tab3.after(0, loading_label_tab3.config, {"text": "Processing completed"})

    #  -------------------------------------------------------------------------------------------------------------

def tab3_draw_audio(audio_path):
    global plot_canvas_tab3
    
    # Clear existing plot
    for widget in plot_frame_tab3.winfo_children():
        widget.destroy()
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot the waveplot
    audio_name = os.path.basename(audio_path)
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title='Waveplot of ' + audio_name)
    
    # Embed the plot in the Tkinter window
    plot_canvas_tab3 = FigureCanvasTkAgg(fig, master=plot_frame_tab3)
    plot_canvas_tab3.draw()
    plot_canvas_tab3.get_tk_widget().pack(expand=1, fill='both')

def tab3_load_audio(audio_path):
    pygame.init()
    pygame.mixer.music.load(audio_path)
    sound = pygame.mixer.Sound(audio_path)

def tab3_play_stop_audio():
    selected_audio = audio_path.get()
    if not selected_audio:
        messagebox.showwarning("Warning", "Please select an Audio file (.wav .mp3 .flac)")
        log("Warning: No Audio selected")
        return
    
    pygame.init()
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        loading_label_tab3.config(text="Audio Stopped")
    else:
        pygame.mixer.music.load(audio_path.get())
        duration = pygame.mixer.Sound(audio_path.get()).get_length()
        loading_label_tab3.config(text="Playing Audio...")
        pygame.mixer.music.play()
        tab3.after(int(duration*1000), loading_label_tab3.config, {"text": ""})
        
# Process audio files in the selected folder
def tab1_process_audio(folder, plot_type):
    audio_files = [f for f in os.listdir(folder) if f.endswith(('.flac', '.wav', '.mp3', '.ogg'))]
    if not audio_files:
        tab1.after(0, log, "No audio files found in the selected folder")
        tab1.after(0, loading_label_tab1.config, {"text": ""})
        return

    tab1.after(0, log, f"Found {len(audio_files)} audio files")

    for widget in plot_area_tab1.winfo_children():
        widget.destroy()

    for audio_file in audio_files:
        file_path = os.path.join(folder, audio_file)
        y, sr = librosa.load(file_path)
        tab1.after(0, log, f"Processing file: {audio_file}")

        # Plot based on selected plot type
        plt.figure(figsize=(9, 3))
        if plot_type == "Spectrogram":
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram of {audio_file}')
        elif plot_type == "MFCCs":
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            librosa.display.specshow(mfccs, sr=sr, x_axis='time')
            plt.colorbar()
            plt.title(f'MFCCs of {audio_file}')
        elif plot_type == "Chroma Features":
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
            plt.colorbar()
            plt.title(f'Chroma Features of {audio_file}')
        elif plot_type == "Zero-Crossing Rate":
            zero_crossings = librosa.feature.zero_crossing_rate(y)
            plt.plot(zero_crossings[0])
            plt.title(f'Zero-Crossing Rate of {audio_file}')
        elif plot_type == "Frequency Analysis":
            D = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            plt.plot(freqs, np.mean(D, axis=1))
            plt.title(f'Frequency Analysis of {audio_file}')
        
        plt.tight_layout()
        
        # Show the plot in the Tkinter window
        fig = plt.gcf()
        canvas = FigureCanvasTkAgg(fig, master=plot_area_tab1)
        tab1.after(0, canvas.draw)
        tab1.after(0, canvas.get_tk_widget().pack, {"fill": tk.BOTH, "expand": True})
        plt.close()
    
    tab1.after(0, loading_label_tab1.config, {"text": "Processing completed"})
#endregion 

#region UI Initialization
# Main window
root = tk.Tk()
root.title("APTT App")
root.geometry("1200x900")

# Custom Style
style = ttk.Style()

style.theme_create("colored_tabs", parent="alt", settings={
    "TNotebook.Tab": {
        "configure": {
            "padding": [10, 5],
            "background": "#f0f0f0",
            "foreground": "#000000"
        },
        "map": {
            "background": [("selected", "#4caf50"), ("active", "#a5d6a7")],
            "foreground": [("selected", "#ffffff"), ("active", "#000000")]
        }
    }
})

style.theme_use("colored_tabs")

# Create tabs
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')
notebook.bind("<<NotebookTabChanged>>", on_tab_selected)
selected_tab = 0

# Tab 1: Audio Analyzer
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='Audio Analyzer')

# Tab 2: Features Extraction
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text='Features Extraction')

# Tab 3: Audio Playing & Model Testing
tab3 = ttk.Frame(notebook)
notebook.add(tab3, text='Model Testing')
#endregion 

# ===================================================
#
# Tab 1 Audio Analyzer
#
# ===================================================
#region Tab1 UI
# Folder selection
folder_path = tk.StringVar()
browse_button = tk.Button(tab1, text="Browse Folder", command=browse_folder)
browse_button.grid(row=0, column=1, padx=2, pady=10, sticky="e")
folder_entry = tk.Entry(tab1, textvariable=folder_path, width=100)
folder_entry.grid(row=0, column=2, columnspan=4, padx=2, pady=10, sticky="w")

# Dropdown menus
lbl_plot_type = tk.Label(tab1, text="Plot Type:")
lbl_plot_type.grid(row=4, column=1, padx=2, pady=10, sticky="e")
dropdown_var1 = tk.StringVar()
dropdown_var1.set("Spectrogram")
dropdown_menu1 = tk.OptionMenu(tab1, dropdown_var1, "Spectrogram", "MFCCs", "Chroma Features", "Zero-Crossing Rate", "Frequency Analysis")
dropdown_menu1.grid(row=4, column=2, padx=2, pady=10, sticky="w")

# lbl_other = tk.Label(tab1, text="Other Type:")
# lbl_other.grid(row=4, column=3, padx=2, pady=10, sticky="e")
# dropdown_var2 = tk.StringVar()
# dropdown_var2.set("Select an option")
# dropdown_menu2 = tk.OptionMenu(tab1, dropdown_var2, "Option 1", "Option 2", "Option 3")
# dropdown_menu2.grid(row=4, column=4, padx=2, pady=10, sticky="w")

# Submit button
submit_button = tk.Button(tab1, text="Submit", command=submit_tab1, background="mediumseagreen", foreground="white")
submit_button.grid(row=6, column=6, padx=2, pady=10, sticky="e")

# Log area
log_label_tab1 = tk.Label(tab1, text="Logs:")
log_label_tab1.grid(row=8, column=0, padx=30, pady=5)
log_area_tab1 = scrolledtext.ScrolledText(tab1, width=140, height=10, state=tk.DISABLED)
log_area_tab1.grid(row=9, column=1, columnspan=6, padx=2, pady=5)

# Plot area with scrollable frame
plot_label_tab1 = tk.Label(tab1, text="Plot:")
plot_label_tab1.grid(row=10, column=0, padx=30, pady=5)
plot_frame_tab1 = tk.Frame(tab1)
plot_frame_tab1.grid(row=11, column=1, columnspan=6, padx=2, pady=5)
plot_canvas_tab1 = tk.Canvas(plot_frame_tab1, width=980, height=400)  # Set the width and height of the plot canvas
plot_scrollbar_tab1 = ttk.Scrollbar(plot_frame_tab1, orient=tk.VERTICAL, command=plot_canvas_tab1.yview)
plot_area_tab1 = tk.Frame(plot_canvas_tab1)

plot_area_tab1.bind(
    "<Configure>",
    lambda e: plot_canvas_tab1.configure(
        scrollregion=plot_canvas_tab1.bbox("all")
    )
)

plot_canvas_tab1.create_window((0, 0), window=plot_area_tab1, anchor="nw")
plot_canvas_tab1.configure(yscrollcommand=plot_scrollbar_tab1.set)
plot_canvas_tab1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
plot_scrollbar_tab1.pack(side=tk.RIGHT, fill=tk.Y)

# Loading label
loading_label_tab1 = tk.Label(tab1, text="")
loading_label_tab1.grid(row=7, column=0, columnspan=7)
#endregion

# ===================================================
#
# Tab 2 Features Extraction
#
# ===================================================
#region Tab2 UI
# Database selection
database_path = tk.StringVar()
database_browse_button = tk.Button(tab2, text="Browse Database Folder", command=browse_folder)
database_browse_button.grid(row=0, column=1, padx=2, pady=10, sticky="e")
database_entry = tk.Entry(tab2, textvariable=database_path, width=100)
database_entry.grid(row=0, column=2, columnspan=4, padx=2, pady=10, sticky="w")

# # Check buttons
# var1_show_graphs = tk.IntVar()
# chkbtn_show_graphs = tk.Checkbutton(tab2, text="Show Graphs", variable=var1_show_graphs)
# chkbtn_show_graphs.grid(row=1, column=1, padx=2, pady=10, sticky="w")

# # Scale
# scale_maxiter_var = tk.IntVar(value=1)
# scale_label = tk.Label(tab2, text="Select Max Iteration:")
# scale_label.grid(row=1, column=2, padx=10, pady=10, sticky=tk.W)

# scale = tk.Scale(tab2, from_=1, to=10000, orient=tk.HORIZONTAL, variable=scale_maxiter_var, command=update_scale_value)
# scale.grid(row=1, column=3, padx=10, pady=10, sticky=tk.EW)

# scale_value_label = tk.Label(tab2, text=f"Max Iteration: {scale_maxiter_var.get()}")
# scale_value_label.grid(row=1, column=4, columnspan=2, padx=10, pady=10)

# Dropdown menus
lbl_mode_type = tk.Label(tab2, text="App Mode:")
lbl_mode_type.grid(row=3, column=1, padx=2, pady=10, sticky="e")
dropdown_mode = tk.StringVar(tab2)
dropdown_mode.set("extract")
dropdown_mode_menu1 = tk.OptionMenu(tab2, dropdown_mode, "extract")
dropdown_mode_menu1.grid(row=3, column=2, padx=2, pady=10, sticky="w")

lbl_speaker = tk.Label(tab2, text="Select Speaker:")
lbl_speaker.grid(row=3, column=3, padx=10, pady=10, sticky="e")
dropdown_speaker_var2 = tk.StringVar(tab2)
dropdown_speaker_var2.set("**ALL**")
dropdown_speaker_menu2 = tk.OptionMenu(tab2, dropdown_speaker_var2, "**ALL**")
dropdown_speaker_menu2.grid(row=3, column=4, padx=10, pady=10, sticky="w")

# lblPlotType = tk.Label(tab2, text="Plot Type:")
# lblPlotType.grid(row=4, column=1, padx=2, pady=10, sticky="w")
# dropdown_var3 = tk.StringVar()
# dropdown_var3.set("Select an option")
# dropdown_menu3 = tk.OptionMenu(tab2, dropdown_var3, "Option 1", "Option 2", "Option 3")
# dropdown_menu3.grid(row=4, column=3, padx=2, pady=10, sticky="e")


# Submit button
submit_button = tk.Button(tab2, text="Start", command=submit_tab2, background="mediumseagreen", foreground="white")
submit_button.grid(row=6, column=6, padx=2, pady=10, sticky="e")

# Log area
log_label_tab2 = tk.Label(tab2, text="Logs:")
log_label_tab2.grid(row=8, column=0, padx=30, pady=5)
log_area_tab2 = scrolledtext.ScrolledText(tab2, width=140, height=10, state=tk.DISABLED)
log_area_tab2.grid(row=9, column=1, columnspan=6, padx=2, pady=5)

# Plot area with scrollable frame
plot_label = tk.Label(tab2, text="Plot:")
plot_label.grid(row=10, column=0, padx=30, pady=5)
plot_frame = tk.Frame(tab2)
plot_frame.grid(row=11, column=1, columnspan=6, padx=2, pady=5)
plot_canvas = tk.Canvas(plot_frame, width=980, height=400)  # Set the width and height of the plot canvas
plot_scrollbar = ttk.Scrollbar(plot_frame, orient=tk.VERTICAL, command=plot_canvas.yview)
plot_area = tk.Frame(plot_canvas)

plot_area.bind(
    "<Configure>",
    lambda e: plot_canvas.configure(
        scrollregion=plot_canvas.bbox("all")
    )
)

plot_canvas.create_window((0, 0), window=plot_area, anchor="nw")
plot_canvas.configure(yscrollcommand=plot_scrollbar.set)
plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
plot_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Loading label
loading_label_tab2 = tk.Label(tab2, text="")
loading_label_tab2.grid(row=7, column=0, columnspan=7)
#endregion

# ===================================================
#
# Tab 3 Audio Playing & Model Testing
#
# ===================================================
#region Tab3 UI

# # frame for the buttons
# frame = tk.Frame(tab3)
# frame.pack(pady=20)

# Files selection
audio_path = tk.StringVar()
audio_browse_button = tk.Button(tab3, text="Browse Audio File", command=browse_audio)
audio_browse_button.grid(row=0, column=1, padx=2, pady=10, sticky="e")
audio_entry = tk.Entry(tab3, textvariable=audio_path, width=100)
audio_entry.grid(row=0, column=2, columnspan=4, padx=2, pady=10, sticky="w")

model_path = tk.StringVar()
model_browse_button = tk.Button(tab3, text="Browse Model File", command=browse_model)
model_browse_button.grid(row=1, column=1, padx=2, pady=10, sticky="e")
model_entry = tk.Entry(tab3, textvariable=model_path, width=100)
model_entry.grid(row=1, column=2, columnspan=4, padx=2, pady=10, sticky="w")

features_path = tk.StringVar()
features_browse_button = tk.Button(tab3, text="Browse Features File", command=browse_features)
features_browse_button.grid(row=2, column=1, padx=2, pady=10, sticky="e")
features_entry = tk.Entry(tab3, textvariable=features_path, width=100)
features_entry.grid(row=2, column=2, columnspan=4, padx=2, pady=10, sticky="w")

# Play button
tab3_play_button = tk.Button(tab3, text="Play/Stop", command=tab3_play_stop_audio, background="mediumseagreen", foreground="white")
tab3_play_button.grid(row=0, column=2, columnspan=4, padx=2, pady=10, sticky="e")

# Submit button
submit_button = tk.Button(tab3, text="Analyse", command=submit_tab3, background="mediumseagreen", foreground="white")
submit_button.grid(row=6, column=6, padx=2, pady=10, sticky="e")

# # label to display the information
# tab3_info_label = tk.Label(tab3, text="", pady=10)
# tab3_info_label.pack()

# Log area
log_label_tab3 = tk.Label(tab3, text="Logs:")
log_label_tab3.grid(row=8, column=0, padx=30, pady=5)
log_area_tab3 = scrolledtext.ScrolledText(tab3, width=140, height=10, state=tk.DISABLED)
log_area_tab3.grid(row=9, column=1, columnspan=6, padx=2, pady=5)

# Plot area with scrollable frame
plot_label_tab3 = tk.Label(tab3, text="Plot:")
plot_label_tab3.grid(row=10, column=0, padx=30, pady=5)
plot_frame_tab3 = tk.Frame(tab3)
plot_frame_tab3.grid(row=11, column=1, columnspan=6, padx=2, pady=5)
plot_canvas_tab3 = tk.Canvas(plot_frame_tab3, width=980, height=400)  # Set the width and height of the plot canvas
# plot_scrollbar_tab3 = ttk.Scrollbar(plot_frame_tab3, orient=tk.VERTICAL, command=plot_canvas_tab3.yview)
plot_area_tab3 = tk.Frame(plot_canvas_tab3)

plot_area.bind(
    "<Configure>",
    lambda e: plot_canvas.configure(
        scrollregion=plot_canvas.bbox("all")
    )
)

plot_canvas_tab3.create_window((0, 0), window=plot_area_tab3, anchor="nw")
# plot_canvas_tab3.configure(yscrollcommand=plot_scrollbar_tab3.set)
plot_canvas_tab3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
# plot_scrollbar_tab3.pack(side=tk.RIGHT, fill=tk.Y)

# Loading label
loading_label_tab3 = tk.Label(tab3, text="")
loading_label_tab3.grid(row=7, column=0, columnspan=7)
#endregion

# Start the GUI event loop
root.mainloop()
