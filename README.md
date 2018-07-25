# Hackathon 2018

location: https://garage-06.visualstudio.com/Communication%20360%C2%B0%201214%2074065/_git/Communication%20360%C2%B0%201214%2074065

Communication 360°(Making communication more efficient through AI)

SmartOn Email
Running server for email analysis and improvements
Run `python3 run.py` to run the server locally!

SmartOn Emotion
Identifying emotions expressed by each face detected in the frame.
Run `ipython3 emotion.py` to check the realtime video emotions

Make sure these Dependencies are present or Install Anaconda in windows and add all these libraries using the commands like:
conda install -c conda-forge flask-oauthlib

python3 - use pip (pip3) to install the following:
- flask
- flask_oauthlib
- requests
- pickle
- nltk - use nltk.download() to install the following: 
    - punkt
    - subjectivity
    - vader_lexicon
    - maxent_ne_chunker (for clean.py only)
    - averaged_perceptron_tagger (for clean.py only)
