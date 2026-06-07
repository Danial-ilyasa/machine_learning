# simulasi jaringan saraf tiruan
import numpy as np
import random
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class JaringanSyarafTiruan:
    def __init__(self, size):
        self.jum_layer = len(size)
        self.ukuran = size
        self.bias = [np.random.randn(y, 1) for y in size[1:]]
        self.bobot = [np.random.randn(y, x) for x, y in zip(size[:-1], size[1:])]
    
    def umpan_maju(self, a):
        for b, w in zip(self.bias, self.bobot):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a
    
    def evaluate(self, data_uji):
        benar = 0
        for x, y in data_uji:
            output = self.umpan_maju(x)
            if np.argmax(output) == np.argmax(y):
                benar += 1
        return benar

    def belajar(self, data_latih, masa_belajar, ukuran_batch, learning_rate, data_uji=None):
        if data_uji:
            n_uji = len(data_uji)
        n = len(data_latih)
        for x in range(masa_belajar):
            random.shuffle(data_latih)
            mini_batch = [data_latih[k:k + ukuran_batch] for k in range(0, n, ukuran_batch)]
            for mini_batch in mini_batch:
                self.update_batch(mini_batch, learning_rate)
            if data_uji:
               print(f"Epoch {x}: {self.evaluate(data_uji)} / {n_uji}")
            else:
                print(f"Epoch {x} selesai")
    
    def update_batch(self, mini_batch, learning_rate):
        x_bias = [np.zeros(b.shape) for b in self.bias]
        x_bobot = [np.zeros(w.shape) for w in self.bobot]
        for x, y in mini_batch:
            delta_x_bias, delta_x_bobot = self.propagasi_mundur(x, y)
            x_bias = [xb + dxb for xb, dxb in zip(x_bias, delta_x_bias)]
            x_bobot = [xw + dxw for xw, dxw in zip(x_bobot, delta_x_bobot)]
        self.bias = [b - (learning_rate / len(mini_batch)) * xb for b, xb in zip(self.bias, x_bias)]
        self.bobot = [w - (learning_rate / len(mini_batch)) * xw for w, xw in zip(self.bobot, x_bobot)]

    def propagasi_mundur(self, x, y):
        x_bias = [np.zeros(b.shape) for b in self.bias]
        x_bobot = [np.zeros(w.shape) for w in self.bobot]
        aktivasi = x
        daftar_aktivasi = [x]
        daftar_z = []
        for b, w in zip(self.bias, self.bobot):
            z = np.dot(w, aktivasi) + b
            daftar_z.append(z)
            aktivasi = sigmoid(z)
            daftar_aktivasi.append(aktivasi)
        delta = self.cost_derivative(daftar_aktivasi[-1], y) * sigmoid_prime(daftar_z[-1])
        x_bias[-1] = delta
        x_bobot[-1] = np.dot(delta, daftar_aktivasi[-2].T)
        for l in range(2, self.jum_layer):
            z = daftar_z[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.bobot[-l + 1].T, delta) * sp
            x_bias[-l] = delta
            x_bobot[-l] = np.dot(delta, daftar_aktivasi[-l - 1].T)
        return (x_bias, x_bobot)

    
    def cost_derivative(self, output_aktivasi, y):
        return output_aktivasi - y
    
    def simpan(self, nama_file):
        data = {
            "jum_layer": self.jum_layer,
            "ukuran": self.ukuran,
            "bias": self.bias,
            "bobot": self.bobot
        }
        with open(nama_file, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def muat(nama_file):
        with open(nama_file, 'rb') as f:
            data = pickle.load(f)
        jaringan = JaringanSyarafTiruan(data["ukuran"])
        jaringan.jum_layer = data["jum_layer"]
        jaringan.bias = [np.array(b) for b in data["bias"]]
        jaringan.bobot = [np.array(w) for w in data["bobot"]]
        return jaringan