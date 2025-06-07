import numpy as np
import mediapipe as mp
import cv2
import scipy.signal as signal

# Inisialisasi MediaPipe FaceDetection di global supaya tidak dibuat berulang
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def cpu_POS(signal, **kargs):
    """
    POS (Plane-Orthogonal-to-Skin) method untuk ekstraksi sinyal rPPG.

    Params:
        signal: Numpy array dengan bentuk [#estimators, 3 (RGB), #frames]
        fps: frame per second dari video yang digunakan

    Return:
        H: Sinyal rPPG hasil POS method untuk setiap estimator (baris)
    """
    eps = 1e-9
    X = signal
    e, c, f = X.shape
    w = int(1.6 * kargs['fps'])

    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    H = np.zeros((e, f))

    for n in range(w, f):
        m = n - w + 1
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2) + eps)
        M = np.expand_dims(M, axis=2)
        Cn = np.multiply(M, Cn)

        S = np.einsum('eij,ejk->eik', Q, Cn)  # Dot product batch e
        # Hanya estimator pertama biasanya, jadi ambil index 0
        S = S[0]
        # Swap axis ke [window, 2]
        S = np.swapaxes(S, 0, 1)

        S1 = S[:, 0]
        S2 = S[:, 1]
        alpha = np.std(S1) / (eps + np.std(S2))
        Hn = S1 + alpha * S2
        Hnm = Hn - np.mean(Hn)

        H[:, m:(n + 1)] += Hnm

    return H

def get_rgb_roi(frame):
    """
    Deteksi wajah dan ekstraksi rata-rata nilai RGB dari ROI.

    Return:
        r_signal, g_signal, b_signal
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    if results.detections:
        # Ambil deteksi pertama
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
        bw, bh = int(bbox.width * w), int(bbox.height * h)

        # Batasi koordinat agar tidak keluar frame
        x = max(0, x)
        y = max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)

        roi = frame[y:y+bh, x:x+bw]

        if roi.size == 0:
            return None, None, None

        r = np.mean(roi[:, :, 2])
        g = np.mean(roi[:, :, 1])
        b = np.mean(roi[:, :, 0])
        return r, g, b
    else:
        return None, None, None

def get_rppg(r, g, b, fps=30):
    """
    Proses sinyal rPPG dari sinyal R, G, B.

    Return:
        rppg_signal: array sinyal rPPG
        heart_rate: estimasi heart rate (BPM)
        peaks: index peak sinyal
        hrv: variabel tambahan (tidak dipakai)
    """
    if len(r) < 60:
        # Data belum cukup
        return np.array([]), 0, [], None

    r = np.array(r)
    g = np.array(g)
    b = np.array(b)

    # Bentuk array untuk POS
    X = np.stack([r, g, b], axis=1).T[np.newaxis, ...]  # bentuk (1, 3, n)

    # Hitung rPPG dengan POS
    rppg = cpu_POS(X, fps=fps)[0]

    # Filter bandpass 0.7 - 3.5 Hz (42 - 210 BPM)
    b, a = signal.butter(3, [0.7/(fps/2), 3.5/(fps/2)], btype='band')
    rppg = signal.filtfilt(b, a, rppg)

    # Normalisasi sinyal
    rppg = (rppg - np.min(rppg)) / (np.max(rppg) - np.min(rppg) + 1e-9)

    # Cari peak
    peaks, _ = signal.find_peaks(rppg, distance=fps*0.5)

    if len(peaks) > 1:
        # Hitung heart rate (BPM)
        rr_intervals = np.diff(peaks) / fps
        heart_rate = 60 / np.mean(rr_intervals)
    else:
        heart_rate = 0

    return rppg, heart_rate, peaks, None
