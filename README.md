### Notes
- This repo only contain partial audio data, for full audio files you may download from [A Dataset for Voice-Based Human Identity Recognition](https://data.mendeley.com/datasets/zw4p4p7sdh/1)
- To run the notebook correctly, you may need jupyter. Simply use the command `pip install -r requirements.txt` in your terminal. This will install all the packages required and listed in from the requirements.txt

### Generate the Executable
- To generate executable file for the application, install `pip install pyinstaller`. 
- Then, use the command `pyinstaller --name=APTTApp --onefile --OPR001_Audio_Analyser_train_test_App.py` to generate the executable. 
- To run the executable successfully, you must copy the folder `packages` together with the App Executable in the same location. 

### Notebooks
- This project is part of [Pattern Recognition Project on Google Colab](https://colab.research.google.com/drive/1N1XEbZBIv7_ScIm4zEa8RT3AmqHi7GA4?usp=sharing)

### References
- [Sound Classification by Juan Pablo Bello](https://s18798.pcdn.co/jpbello/wp-content/uploads/sites/1691/2018/01/8-classification.pdf)