import os
import pandas as pd

audio_files = os.listdir('data/test_data/audio/')
filenames = [os.path.splitext(file)[0] for file in audio_files if file.endswith('.mp3')]

df = pd.DataFrame(filenames, columns=['Filename'])

for column in ['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']:
    df[column] = 0.0

csv_path = 'data/test_split.csv'

df.to_csv(csv_path, index=False)

