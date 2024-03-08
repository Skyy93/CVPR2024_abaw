import numpy as np
import onnxruntime as ort
from PIL import Image
import urllib.request
from pathlib import Path
import pandas as pd
import os
from tqdm.contrib.concurrent import process_map

input = 'data/raw_face_aligned'
output = 'data/raw_libreface'

if not Path('au_enc.onnx').exists():
    urllib.request.urlretrieve(
        'https://github.com/ihp-lab/OpenSense/raw/master/Utilities/LibreFace/LibreFace_AU_Encoder.onnx',
        'au_enc.onnx')
if not Path('au_reg.onnx').exists():
    urllib.request.urlretrieve(
        'https://github.com/ihp-lab/OpenSense/raw/master/Utilities/LibreFace/LibreFace_AU_Intensity.onnx',
        'au_reg.onnx')
if not Path('au_det.onnx').exists():
    urllib.request.urlretrieve(
        'https://github.com/ihp-lab/OpenSense/raw/master/Utilities/LibreFace/LibreFace_AU_Presence.onnx', 'au_det.onnx')
if not Path('fer.onnx').exists():
    urllib.request.urlretrieve('https://github.com/ihp-lab/OpenSense/raw/master/Utilities/LibreFace/LibreFace_FE.onnx',
                               'fer.onnx')

detection = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
regression = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
emotions = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
labels = [f'AU{str(x)}_c' for x in detection]
labels.extend([f'AU{str(x)}_r' for x in regression])
labels.extend(emotions)

au_enc = ort.InferenceSession('au_enc.onnx', providers=['CPUExecutionProvider'])
au_det = ort.InferenceSession('au_det.onnx', providers=['CPUExecutionProvider'])
au_reg = ort.InferenceSession('au_reg.onnx', providers=['CPUExecutionProvider'])
fer = ort.InferenceSession('fer.onnx', providers=['CPUExecutionProvider'])


def method(folder):
    images_preprocessed = []
    for file in sorted(folder.glob('*.jpg')):
        image = np.array(Image.open(file).resize((224, 224), Image.LANCZOS))
        image = image.astype(float)
        image = image / 255.0
        # supposedly RGB
        image = image - [[[0.485, 0.456, 0.406]]]
        image = image / [[[0.229, 0.224, 0.225]]]
        image = image.astype(np.float32)
        image = image.transpose((2, 0, 1))
        images_preprocessed.append(image)

    if len(images_preprocessed) == 0:
        print(f'empty: {folder}')
        return

    images_preprocessed_encoded = [au_enc.run(['feature'], {'image': np.expand_dims(x, axis=0)})
                                   for x in
                                   images_preprocessed]

    preds = [[au_det.run(['au_presence'], {'feature': np.expand_dims(np.squeeze(x), axis=0)})
              for x in images_preprocessed_encoded],
             [au_reg.run(['au_intensity'], {'feature': np.expand_dims(np.squeeze(x), axis=0)})
              for x in images_preprocessed_encoded],
             [fer.run(['FEs', 'onnx::Gemm_204'], {'image': np.expand_dims(x, axis=0)})
              for x in images_preprocessed]]

    preds, fer_feats = np.concatenate(list((*preds[:2], [[[y]] for x in preds[2] for y in x[0]])),
                                      axis=-1).squeeze(), np.concatenate([[y] for x in preds[2] for y in x[1]])

    try:
        df = pd.DataFrame(preds, columns=labels)
    except ValueError:
        print(f'one image: {folder}')
        df = pd.DataFrame([preds], columns=labels)

    df['fer_feats'] = [x for x in fer_feats]
    df.to_pickle(Path(output) / (folder.name + '.xz'))


folders = list(Path(input).glob('*'))
process_map(method, folders, max_workers=int(os.cpu_count() / 2), chunksize=2)
