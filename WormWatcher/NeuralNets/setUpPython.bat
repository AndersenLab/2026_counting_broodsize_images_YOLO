
cd "C:\Program Files\Python39\Scripts"

::install python dependencies
pip install numpy opencv-python pandas natsort pillow sendgrid easygui pypylon LabJackPython statsmodels matplotlib seaborn psutil bs4 openpyxl XlsxWriter scipy scikit-learn
pip3 install torch torchvision torchaudio ultralytics

::remove cmd /k to quit cleanly at end
cmd /k