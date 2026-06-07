from pathlib import Path
import numpy as np
from sistem import JaringanSyarafTiruan

DATA_DIR = Path(__file__).resolve().parents[1] / "jst-01/data"
MODEL_FILE = Path(__file__).resolve().with_name("model.pkl")


def flatten_dataset(images):
    images = np.array(images, dtype=np.float32)
    if images.ndim == 1:
        return images.reshape(-1, 1)
    return images.reshape(images.shape[0], -1)

def flatten_sample(image):
    image = np.array(image, dtype=np.float32)
    return image.reshape(-1, 1)

def one_hot(labels, num_classes=10):
    labels = np.array(labels, dtype=np.int64).reshape(-1)
    if labels.ndim != 1:
        raise ValueError("Labels harus berupa vektor 1-dimensi")
    y = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    y[np.arange(labels.shape[0]), labels] = 1.0
    return [vec.reshape(-1, 1) for vec in y]


yyuu = 4
size = [784, 64, 16, 10]

if __name__ == "__main__":
    train_images = np.load(DATA_DIR / "ti.npy")
    train_labels = np.load(DATA_DIR / "tl.npy")

    test_images = np.load(DATA_DIR / "test_images.npy")
    test_labels = np.load(DATA_DIR / "test_labels.npy")

    x_train = flatten_dataset(train_images) / 255.0
    y_train = one_hot(train_labels, num_classes=10)
    data_latih = [(x.reshape(-1, 1), y) for x, y in zip(x_train, y_train)]

    data_uji = list(zip(flatten_dataset(test_images) / 255.0, one_hot(test_labels, num_classes=10)))
    net = JaringanSyarafTiruan(size)
    input_baru = flatten_sample(test_images[yyuu]) / 255.0

    print(f"Label data [{yyuu}]:", test_labels[yyuu])

    net2 = None
    if MODEL_FILE.exists():
        net2 = JaringanSyarafTiruan.muat(str(MODEL_FILE))

    if net2 is not None:
        # net.belajar(data_latih, masa_belajar=1, ukuran_batch=64, learning_rate=1.0)
        output = net2.umpan_maju(input_baru)
        print("Model berhasil dimuat dari", MODEL_FILE)
        print("hasil:\n", output)
        print("Prediksi kelas:", np.argmax(output))
    else:
        net.belajar(data_latih, masa_belajar=30, ukuran_batch=32, learning_rate=0.1)
        output = net.umpan_maju(input_baru)
        print("Model tidak ditemukan, menggunakan model baru hasil pelatihan.")
        print("hasil:\n", output)
        print("Prediksi kelas:", np.argmax(output))
        net.simpan(str(MODEL_FILE))