import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os
import numpy as np
import librosa
import librosa.display
import threading
import subprocess

import sys
import random as rand
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import History

def run_external_script(script_name, *args):
    # Command to run the external script
    command = ['python3', script_name, *args]
    try:
        # Run the external script
        result = subprocess.run(command, capture_output=True, text=True)
        # Display the output in the text area
        log(result.stdout)
        log(result.stderr)
    except Exception as e:
        log(f"Error: {e}\n")

def submit_tab2():
    arg1 = dropdown_mode.get() # mode = "train" # "train" or "analyse" or "test"
    arg2 = dropdown_speaker_var2.get() # speaker = 2 # speaker identity for testing
    arg3 = dropdown_speaker_var2.get() # train_speaker = 1 # speaker identity for training, -1 to ignore
    arg4 = str(scale_maxiter_var.get()) # max_iter = 1 # iteration (usually for training model with random seed)
    arg5 = str(bool(var1_show_graphs.get())) # show_graphs = False
    arg6 = database_entry.get() # dataset_path = f"audio\speakers\{database_name}"
    run_external_script("speech_features_extraction.py", arg1, arg2, arg3, arg4, arg5, arg6)

# Function to browse folder
def browse_folder():
    folder_selected = filedialog.askdirectory()
    if database_path and selected_tab == 1:
        database_path.set(folder_selected)
        update_dropdown(database_path.get())
    else:
        folder_path.set(folder_selected)
    log(f"Selected folder: {folder_selected}")
    
def update_dropdown(folder_path):
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    dropdown_speaker_var2.set(folders[0] if folders else "No folders found")
    menu = dropdown_speaker_menu2['menu']
    menu.delete(0, 'end')
    for folder in folders:
        menu.add_command(label=folder, command=tk._setit(dropdown_speaker_var2, folder))
    # log(f"Folder loaded: {folder_path}\n")
    
# Function to submit form
def submit_form():
    selected_folder = folder_path.get()
    option1 = var1.get()
    option2 = var2.get()
    selected_plot_type = dropdown_var1.get()
    selected_option2 = dropdown_var2.get()
    
    if not selected_folder:
        messagebox.showwarning("Warning", "Please select a folder")
        log("Warning: No folder selected")
    else:
        result = f"Folder: {selected_folder}\nOption 1: {option1}\nOption 2: {option2}\Plot Type: {selected_plot_type}\nDropdown 2: {selected_option2}"
        messagebox.showinfo("Form Submitted", result)
        log(f"Form submitted:\n{result}")
        
        loading_label.config(text="Loading, please wait...")
        threading.Thread(target=process_audio, args=(selected_folder, selected_plot_type)).start()

def update_scale_value(val):
    scale_value_label.config(text=f"Max Iteration: {scale_maxiter_var.get()}")
    
def on_tab_selected(event):
    global selected_tab
    selected_tab = event.widget.index("current")
    # print(f'current tab : {selected_tab}')
    
# Function to log messages to the text area
def log(message):
    log_area.config(state=tk.NORMAL)
    log_area.insert(tk.END, message + "\n")
    log_area.config(state=tk.DISABLED)
    log_area.yview(tk.END)

# Function to process audio files in the selected folder
def process_audio(folder, plot_type):
    audio_files = [f for f in os.listdir(folder) if f.endswith(('.flac', '.wav', '.mp3', '.ogg'))]
    if not audio_files:
        root.after(0, log, "No audio files found in the selected folder")
        root.after(0, loading_label.config, {"text": ""})
        return

    root.after(0, log, f"Found {len(audio_files)} audio files")

    for widget in plot_area.winfo_children():
        widget.destroy()

    for audio_file in audio_files:
        file_path = os.path.join(folder, audio_file)
        y, sr = librosa.load(file_path)
        root.after(0, log, f"Processing file: {audio_file}")

        # Plot based on selected plot type
        plt.figure(figsize=(10, 4))
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
            D = np.abs(librosa.stft(y))**2
            freqs, times, Sx = librosa.reassigned_spectrogram(y, sr=sr)
            plt.pcolormesh(times, freqs, librosa.amplitude_to_db(Sx, ref=np.max))
            plt.colorbar()
            plt.title(f'Frequency Analysis of {audio_file}')
        
        plt.tight_layout()
        
        # Show the plot in the Tkinter window
        fig = plt.gcf()
        canvas = FigureCanvasTkAgg(fig, master=plot_area)
        root.after(0, canvas.draw)
        root.after(0, canvas.get_tk_widget().pack, {"fill": tk.BOTH, "expand": True})
        plt.close()
    
    root.after(0, loading_label.config, {"text": "Processing completed"})

# Create main window
root = tk.Tk()
root.title("Audio Processing UI")
root.geometry("1200x900")

# Create a style object
style = ttk.Style()

# Create a custom theme for the notebook
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

# Use the custom theme
style.theme_use("colored_tabs")

# Create tabs
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both')
notebook.bind("<<NotebookTabChanged>>", on_tab_selected)
selected_tab = 0

# Tab 1: Audio Analyzer
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text='Audio Analyzer')

# Tab 2: Model Training
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text='Model Training')

# ===================================================
#
# Tab 1 Audio Analyzer
#
# ===================================================
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

lbl_other = tk.Label(tab1, text="Other Type:")
lbl_other.grid(row=4, column=3, padx=2, pady=10, sticky="e")
dropdown_var2 = tk.StringVar()
dropdown_var2.set("Select an option")
dropdown_menu2 = tk.OptionMenu(tab1, dropdown_var2, "Option 1", "Option 2", "Option 3")
dropdown_menu2.grid(row=4, column=4, padx=2, pady=10, sticky="w")

# lblPlotType = tk.Label(tab1, text="Plot Type:")
# lblPlotType.grid(row=4, column=1, padx=2, pady=10, sticky="w")
# dropdown_var3 = tk.StringVar()
# dropdown_var3.set("Select an option")
# dropdown_menu3 = tk.OptionMenu(tab1, dropdown_var3, "Option 1", "Option 2", "Option 3")
# dropdown_menu3.grid(row=4, column=3, padx=2, pady=10, sticky="e")


# Submit button
submit_button = tk.Button(tab1, text="Submit", command=submit_form, background="mediumseagreen", foreground="white")
submit_button.grid(row=6, column=6, padx=2, pady=10, sticky="e")

# Log area
log_label = tk.Label(tab1, text="Logs:")
log_label.grid(row=8, column=0, padx=30, pady=5)
log_area = scrolledtext.ScrolledText(tab1, width=140, height=10, state=tk.DISABLED)
log_area.grid(row=9, column=1, columnspan=6, padx=2, pady=5)

# Plot area with scrollable frame
plot_label = tk.Label(tab1, text="Plot:")
plot_label.grid(row=10, column=0, padx=30, pady=5)
plot_frame = tk.Frame(tab1)
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
loading_label = tk.Label(tab1, text="")
loading_label.grid(row=7, column=0, columnspan=7)


# ===================================================
#
# Tab 2 Model Training
#
# ===================================================
# Database selection
database_path = tk.StringVar()
database_browse_button = tk.Button(tab2, text="Browse Database Folder", command=browse_folder)
database_browse_button.grid(row=0, column=1, padx=2, pady=10, sticky="e")
database_entry = tk.Entry(tab2, textvariable=database_path, width=100)
database_entry.grid(row=0, column=2, columnspan=4, padx=2, pady=10, sticky="w")

# Checkbuttons
var1_show_graphs = tk.IntVar()
chkbtn_show_graphs = tk.Checkbutton(tab2, text="Show Graphs", variable=var1_show_graphs)
chkbtn_show_graphs.grid(row=1, column=1, padx=2, pady=10, sticky="w")

# Scale
scale_maxiter_var = tk.IntVar(value=1)
scale_label = tk.Label(tab2, text="Select Max Iteration:")
scale_label.grid(row=1, column=2, padx=10, pady=10, sticky=tk.W)

scale = tk.Scale(tab2, from_=1, to=10000, orient=tk.HORIZONTAL, variable=scale_maxiter_var, command=update_scale_value)
scale.grid(row=1, column=3, padx=10, pady=10, sticky=tk.EW)

scale_value_label = tk.Label(tab2, text=f"Max Iteration: {scale_maxiter_var.get()}")
scale_value_label.grid(row=1, column=4, columnspan=2, padx=10, pady=10)

# Dropdown menus
lbl_mode_type = tk.Label(tab2, text="App Mode:")
lbl_mode_type.grid(row=3, column=1, padx=2, pady=10, sticky="e")
dropdown_mode = tk.StringVar(tab2)
dropdown_mode.set("train")
dropdown_mode_menu1 = tk.OptionMenu(tab2, dropdown_mode, "train", "analyse", "test")
dropdown_mode_menu1.grid(row=3, column=2, padx=2, pady=10, sticky="w")

lbl_speaker = tk.Label(tab2, text="Select Speaker:")
lbl_speaker.grid(row=3, column=3, padx=10, pady=10, sticky="e")
dropdown_speaker_var2 = tk.StringVar(tab2)
dropdown_speaker_menu2 = tk.OptionMenu(tab2, dropdown_speaker_var2, "")
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
log_label = tk.Label(tab2, text="Logs:")
log_label.grid(row=8, column=0, padx=30, pady=5)
log_area = scrolledtext.ScrolledText(tab2, width=140, height=10, state=tk.DISABLED)
log_area.grid(row=9, column=1, columnspan=6, padx=2, pady=5)

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
loading_label = tk.Label(tab2, text="")
loading_label.grid(row=7, column=0, columnspan=7)


# Start the GUI event loop
root.mainloop()
