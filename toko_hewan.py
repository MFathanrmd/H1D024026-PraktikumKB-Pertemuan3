import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os

# 1. Definisi Variabel Input dan Output
barang_terjual = ctrl.Antecedent(np.arange(0, 101, 1), 'barang_terjual')
permintaan = ctrl.Antecedent(np.arange(0, 301, 1), 'permintaan')
harga_item = ctrl.Antecedent(np.arange(0, 100001, 1000), 'harga_item')
profit = ctrl.Antecedent(np.arange(0, 4000001, 100000), 'profit')
stok_makanan = ctrl.Consequent(np.arange(0, 1001, 1), 'stok_makanan')

# 2. Definisi Himpunan Fuzzy (Membership Functions)
# Barang Terjual
barang_terjual['rendah'] = fuzz.trimf(barang_terjual.universe, [0, 0, 50])
barang_terjual['sedang'] = fuzz.trimf(barang_terjual.universe, [30, 50, 70])
barang_terjual['tinggi'] = fuzz.trimf(barang_terjual.universe, [50, 100, 100])

# Permintaan
permintaan['rendah'] = fuzz.trimf(permintaan.universe, [0, 0, 150])
permintaan['sedang'] = fuzz.trimf(permintaan.universe, [100, 150, 200])
permintaan['tinggi'] = fuzz.trimf(permintaan.universe, [200, 300, 300])

# Harga Per Item (Skala 1:10000 di grafik, kita pakai nilai asli)
harga_item['murah'] = fuzz.trimf(harga_item.universe, [0, 0, 40000])
harga_item['sedang'] = fuzz.trimf(harga_item.universe, [30000, 50000, 70000])
harga_item['mahal'] = fuzz.trimf(harga_item.universe, [60000, 100000, 100000])

# Profit (Skala 1:1000000 di grafik, kita pakai nilai asli)
profit['rendah'] = fuzz.trimf(profit.universe, [0, 0, 1500000])
profit['sedang'] = fuzz.trimf(profit.universe, [1000000, 2000000, 3000000])
profit['banyak'] = fuzz.trapmf(profit.universe, [2000000, 2500000, 4000000, 4000000])

# Stok Makanan (Output)
stok_makanan['sedang'] = fuzz.trimf(stok_makanan.universe, [0, 500, 900])
stok_makanan['banyak'] = fuzz.trimf(stok_makanan.universe, [600, 1000, 1000])

# 3. Definisi Aturan Fuzzy
rule1 = ctrl.Rule(barang_terjual['tinggi'] & permintaan['tinggi'] & harga_item['murah'] & profit['banyak'], stok_makanan['banyak'])
rule2 = ctrl.Rule(barang_terjual['tinggi'] & permintaan['tinggi'] & harga_item['murah'] & profit['sedang'], stok_makanan['sedang'])
rule3 = ctrl.Rule(barang_terjual['tinggi'] & permintaan['sedang'] & harga_item['murah'] & profit['sedang'], stok_makanan['sedang'])
rule4 = ctrl.Rule(barang_terjual['sedang'] & permintaan['tinggi'] & harga_item['murah'] & profit['sedang'], stok_makanan['sedang'])
rule5 = ctrl.Rule(barang_terjual['sedang'] & permintaan['tinggi'] & harga_item['murah'] & profit['banyak'], stok_makanan['banyak'])
rule6 = ctrl.Rule(barang_terjual['rendah'] & permintaan['rendah'] & harga_item['sedang'] & profit['sedang'], stok_makanan['sedang'])

# 4. Kontrol Sistem
stok_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
stok_simulasi = ctrl.ControlSystemSimulation(stok_ctrl)

# 5. Input Nilai Sesuai Kasus
stok_simulasi.input['barang_terjual'] = 80
stok_simulasi.input['permintaan'] = 255
stok_simulasi.input['harga_item'] = 25000
stok_simulasi.input['profit'] = 3500000

# 6. Perhitungan (Defuzzifikasi)
stok_simulasi.compute()

import matplotlib.pyplot as plt

# 7. Output Hasil
print(f"Hasil Perhitungan Stok Makanan: {stok_simulasi.output['stok_makanan']:.2f}")

# 8. Visualisasi
# Menampilkan grafik untuk setiap input (4 parameter)
barang_terjual.view()
permintaan.view()
harga_item.view()
profit.view()

# Menampilkan grafik output dengan hasil simulasi (parameter ke-5)
stok_makanan.view(sim=stok_simulasi)
plt.title("Hasil Perhitungan Stok Makanan - Kasus 1")

# Pastikan folder visualisasi tersedia
output_dir = 'visualisasi grafik'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Hanya simpan grafik output ke folder visualisasi grafik
save_path = os.path.join(output_dir, 'stok_petshop_result.png')
plt.savefig(save_path)
print(f"Grafik visualisasi output disimpan di: {save_path}")

# Tampilkan semua jendela grafik (non-blocking)
plt.show(block=False)

# Tahan jendela terminal agar tidak langsung tertutup
input("\nTekan Enter pada terminal untuk keluar dan menutup semua grafik...")
plt.close('all')