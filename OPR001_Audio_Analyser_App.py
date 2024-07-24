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

# Function to browse folder
def browse_folder():
    folder_selected = filedialog.askdirectory()
    folder_path.set(folder_selected)
    log(f"Selected folder: {folder_selected}")

# Function to submit form
def submit_form():
    selected_folder = folder_path.get()
    option1 = var1.get()
    option2 = var2.get()
    option3 = var3.get()
    option4 = var4.get()
    selected_plot_type = dropdown_var1.get()
    selected_option2 = dropdown_var2.get()
    selected_option3 = dropdown_var3.get()
    
    if not selected_folder:
        messagebox.showwarning("Warning", "Please select a folder")
        log("Warning: No folder selected")
    else:
        result = f"Folder: {selected_folder}\nOption 1: {option1}\nOption 2: {option2}\Plot Type: {selected_plot_type}\nDropdown 2: {selected_option2}\nDropdown 3: {selected_option3}"
        messagebox.showinfo("Form Submitted", result)
        log(f"Form submitted:\n{result}")
        
        loading_label.config(text="Loading, please wait...")
        threading.Thread(target=process_audio, args=(selected_folder, selected_plot_type)).start()

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

# Folder selection
folder_path = tk.StringVar()
browse_button = tk.Button(root, text="Browse Folder", command=browse_folder)
browse_button.grid(row=0, column=1, padx=2, pady=10, sticky="e")
folder_entry = tk.Entry(root, textvariable=folder_path, width=100)
folder_entry.grid(row=0, column=2, columnspan=4, padx=2, pady=10, sticky="w")

# Checkbuttons
var1 = tk.IntVar()
var2 = tk.IntVar()
var3 = tk.IntVar()
var4 = tk.IntVar()
checkbutton1 = tk.Checkbutton(root, text="Option 1", variable=var1)
checkbutton1.grid(row=2, column=1, padx=2, pady=10)
checkbutton2 = tk.Checkbutton(root, text="Option 2", variable=var2)
checkbutton2.grid(row=2, column=2, padx=2, pady=10)
checkbutton3 = tk.Checkbutton(root, text="Option 3", variable=var3)
checkbutton3.grid(row=2, column=3, padx=2, pady=10)
checkbutton4 = tk.Checkbutton(root, text="Option 4", variable=var4)
checkbutton4.grid(row=2, column=4, padx=2, pady=10)

# Dropdown menus
dropdown_var1 = tk.StringVar()
dropdown_var1.set("Spectrogram")
dropdown_menu1 = tk.OptionMenu(root, dropdown_var1, "Spectrogram", "MFCCs", "Chroma Features", "Zero-Crossing Rate", "Frequency Analysis")
dropdown_menu1.grid(row=3, column=1, padx=2, pady=10)

dropdown_var2 = tk.StringVar()
dropdown_var2.set("Select an option")
dropdown_menu2 = tk.OptionMenu(root, dropdown_var2, "Option 1", "Option 2", "Option 3")
dropdown_menu2.grid(row=3, column=2, padx=2, pady=10)

dropdown_var3 = tk.StringVar()
dropdown_var3.set("Select an option")
dropdown_menu3 = tk.OptionMenu(root, dropdown_var3, "Option 1", "Option 2", "Option 3")
dropdown_menu3.grid(row=3, column=3, padx=2, pady=10)


# Submit button
submit_button = tk.Button(root, text="Submit", command=submit_form, background="mediumseagreen", foreground="white")
submit_button.grid(row=5, column=6, padx=2, pady=10, sticky="w")

# Log area
log_label = tk.Label(root, text="Logs:")
log_label.grid(row=6, column=0, padx=2, pady=5)
log_area = scrolledtext.ScrolledText(root, width=140, height=10, state=tk.DISABLED)
log_area.grid(row=7, column=0, columnspan=7, padx=2, pady=5)

# Plot area with scrollable frame
plot_label = tk.Label(root, text="Plot:")
plot_label.grid(row=8, column=0, padx=2, pady=5)
plot_frame = tk.Frame(root)
plot_frame.grid(row=9, column=0, columnspan=7, padx=2, pady=5)
plot_canvas = tk.Canvas(plot_frame, width=1150, height=400)  # Set the width and height of the plot canvas
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
loading_label = tk.Label(root, text="")
loading_label.grid(row=8, column=0, columnspan=7)

# Start the GUI event loop
root.mainloop()
