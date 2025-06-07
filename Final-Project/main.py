import cv2
import matplotlib.pyplot as plt
from resp import get_resp
import time

def main():
    cap = cv2.VideoCapture(0)  # Webcam
    if not cap.isOpened():
        print("Error: Kamera tidak dapat dibuka")
        return

    start_time = time.time()
    resp_values = []
    time_values = []

    plt.ion()  # Mode interaktif untuk update grafik realtime
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-')
    ax.set_xlabel('Waktu (detik)')
    ax.set_ylabel('Resp')
    ax.set_title('Grafik Sinyal Resp')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Jangan flip frame supaya posisi tidak mirror
        resp, t = get_resp(frame, start_time)

        # Simpan data untuk plot
        resp_values.append(resp)
        time_values.append(t)

        # Update plot
        line.set_xdata(time_values)
        line.set_ydata(resp_values)
        ax.relim()            # Recalculate limits
        ax.autoscale_view()   # Autoscale

        plt.pause(0.001)      # Pause sebentar agar grafik update

        # Tampilkan frame webcam asli (tanpa flip)
        cv2.imshow("Webcam (tidak mirror)", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
