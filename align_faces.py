from pathlib import Path
import mediapipe as mp
from multiprocessing import Pool
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
import imageio_ffmpeg

input = 'data/raw'
output_dimension = 256
output_face = 'data/raw_face'
output_aligned = 'data/raw_face_aligned'
#output_landmarks = 'data/raw_face_landmarks'
min_confidence = 0.5
fps = 5

def image_align(img, face_landmarks, output_size=output_dimension,
                transform_size=4096, enable_padding=True, x_scale=1,
                y_scale=1, em_scale=0.1, alpha=False, pad_mode='const'):
    # img = my_draw_image_by_points(img, face_landmarks[36:60], 1, (0,255,0))
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    # import PIL.Image
    # import scipy.ndimage
    # print(img.size)
    img = Image.fromarray(img)
    lm = np.array(face_landmarks)
    lm[:, 0] *= img.size[0]
    lm[:, 1] *= img.size[1]
    # lm_chin          = lm[0  : 17]  # left-right
    # lm_eyebrow_left  = lm[17 : 22]  # left-right
    # lm_eyebrow_right = lm[22 : 27]  # left-right
    # lm_nose          = lm[27 : 31]  # top-down
    # lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_right = lm[0:16]
    lm_eye_left = lm[16:32]
    lm_mouth_outer = lm[32:]
    # lm_mouth_inner   = lm[60 : 68]  # left-clockwise
    lm_mouth_outer_x = lm_mouth_outer[:, 0].tolist()
    left_index = lm_mouth_outer_x.index(min(lm_mouth_outer_x))
    right_index = lm_mouth_outer_x.index(max(lm_mouth_outer_x))
    # print(left_index,right_index)
    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    # eye_left[[0,1]] = eye_left[[1,0]]
    eye_right = np.mean(lm_eye_right, axis=0)
    # eye_right[[0,1]] = eye_right[[1,0]]
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    # print(lm_mouth_outer)s
    mouth_avg = (lm_mouth_outer[left_index, :] + lm_mouth_outer[right_index, :]) / 2.0
    # mouth_avg[[0,1]] = mouth_avg[[1,0]]

    eye_to_mouth = mouth_avg - eye_avg
    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= x_scale
    y = np.flipud(x) * [-y_scale, y_scale]
    c = eye_avg + eye_to_mouth * em_scale
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    crop_img = img.copy()
    crop_img = crop_img.resize((output_size, output_size), Image.Resampling.LANCZOS)

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        if pad_mode == 'const':
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'constant', constant_values=0)
        else:
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.uint8(np.clip(np.rint(img), 0, 255))
        if alpha:
            mask = 1 - np.clip(3.0 * mask, 0.0, 1.0)
            mask = np.uint8(np.clip(np.rint(mask * 255), 0, 255))
            img = np.concatenate((img, mask), axis=2)
            img = Image.fromarray(img, 'RGBA')
        else:
            img = Image.fromarray(img, 'RGB')
        quad += pad[:2]

    img = img.transform((transform_size, transform_size), Image.Transform.QUAD,
                        (quad + 0.5).flatten(), Image.Resampling.BILINEAR)
    out_image = img.resize((output_size, output_size), Image.Resampling.LANCZOS)

    return crop_img, out_image

if __name__ == '__main__':
    def f(file: Path):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    refine_landmarks=True,
                    max_num_faces=1,
                    min_detection_confidence=min_confidence,
                    min_tracking_confidence=min_confidence)

        Left_eye = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
        Right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
        Lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88,
                178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]

        (Path(output_face) / file.stem).mkdir(parents=True, exist_ok=True)
        (Path(output_aligned) / file.stem).mkdir(parents=True, exist_ok=True)
        #landmarks_fm = []
        reader = imageio_ffmpeg.read_frames(file, output_params=['-vf', f'fps={fps}'])
        meta = next(reader)
        for i, image in enumerate(reader):
            image = np.array(Image.frombytes('RGB', meta['source_size'], image))
            crop_img_path = Path(output_face) / file.stem / (str(i).zfill(5) + '.jpg')
            aligned_img_path = Path(output_aligned) / file.stem / (str(i).zfill(5) + '.jpg')

            results = face_mesh.process(image)
            if results is None or not results.multi_face_landmarks:
                print(f'No face mesh detected for {aligned_img_path}')
                continue
                aligned_image = Image.fromarray(np.zeros((output_dimension, output_dimension, 3), dtype=np.uint8))
                crop = aligned_image
                #landmark_fm = np.zeros((478, 3), dtype=np.float64)
            else:
                face_landmarks = results.multi_face_landmarks[0]
                lm_left_eye_x = []
                lm_left_eye_y = []
                lm_right_eye_x = []
                lm_right_eye_y = []
                lm_lips_x = []
                lm_lips_y = []
                for i in Left_eye:
                    lm_left_eye_x.append(face_landmarks.landmark[i].x)
                    lm_left_eye_y.append(face_landmarks.landmark[i].y)
                for i in Right_eye:
                    lm_right_eye_x.append(face_landmarks.landmark[i].x)
                    lm_right_eye_y.append(face_landmarks.landmark[i].y)
                for i in Lips:
                    lm_lips_x.append(face_landmarks.landmark[i].x)
                    lm_lips_y.append(face_landmarks.landmark[i].y)
                lm_x = lm_left_eye_x + lm_right_eye_x + lm_lips_x
                lm_y = lm_left_eye_y + lm_right_eye_y + lm_lips_y
                landmark_lf = np.array([lm_x, lm_y]).T
                crop, aligned_image = image_align(image, landmark_lf)
                #landmark_fm = [[x.x, x.y, x.z] for x in [y for y in face_landmarks.landmark]]

            #landmarks_fm.append(landmark_fm)
            crop.save(crop_img_path)
            aligned_image.save(aligned_img_path)

        #np.save(Path(output_landmarks) / (file.stem + '.npy'), np.array(landmarks_fm))
        face_mesh.close()


    Path(output_aligned).mkdir(parents=True, exist_ok=True)
    #Path(output_landmarks).mkdir(parents=True, exist_ok=True)
    files = list(Path(input).glob('*.mp4'))
    #debug
    #f(Path('data/raw/04595.mp4'))
    with Pool() as p:
        work = p.imap_unordered(f, files)
        list(tqdm(work, total=len(files)))
