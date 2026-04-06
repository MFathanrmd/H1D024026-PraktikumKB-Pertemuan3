import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
import pandas as pd
import matplotlib.pyplot as plt

# Definisi Variabel
informasi = ctrl.Antecedent(np.arange(0, 101, 1), 'informasi')
persyaratan = ctrl.Antecedent(np.arange(0, 101, 1), 'persyaratan')
petugas = ctrl.Antecedent(np.arange(0, 101, 1), 'petugas')
sarpras = ctrl.Antecedent(np.arange(0, 101, 1), 'sarpras')
kepuasan = ctrl.Consequent(np.arange(0, 401, 1), 'kepuasan')

# Membership Function
for var in [informasi, persyaratan, petugas, sarpras]:
    var['tidak'] = fuzz.trapmf(var.universe, [0, 0, 60, 75])
    var['cukup'] = fuzz.trimf(var.universe, [60, 75, 90])
    var['memuaskan'] = fuzz.trapmf(var.universe, [75, 90, 100, 100])

kepuasan['tidak'] = fuzz.trapmf(kepuasan.universe, [0, 0, 50, 75])
kepuasan['kurang'] = fuzz.trimf(kepuasan.universe, [50, 75, 100])
kepuasan['cukup'] = fuzz.trapmf(kepuasan.universe, [150, 175, 250, 275])
kepuasan['memuaskan'] = fuzz.trapmf(kepuasan.universe, [250, 275, 325, 350])
kepuasan['sangat'] = fuzz.trapmf(kepuasan.universe, [325, 350, 400, 400])

# Mapping
mapping = {
    'Tidak Memuaskan': 'tidak',
    'Cukup Memuaskan': 'cukup',
    'Memuaskan': 'memuaskan',
    'Kurang Memuaskan': 'kurang',
    'Sangat Memuaskan': 'sangat'
}

# Load Rules dari File
rules = []

file_name = '81_rules.csv' 

if os.path.exists(file_name):
    df_rules = pd.read_csv(file_name)
    print("Kolom:", df_rules.columns)

    for _, row in df_rules.iterrows():
        rule = ctrl.Rule(
            informasi[mapping[row['Kejelasan Informasi']]] &
            persyaratan[mapping[row['Kejelasan Persyaratan']]] &
            petugas[mapping[row['Kemampuan Petugas']]] &
            sarpras[mapping[row['Ketersediaan Sarpras']]],
            kepuasan[mapping[row['Kepuasan Pelayanan']]]
        )
        rules.append(rule)

    print(f"Berhasil memuat {len(rules)} aturan.")
else:
    print("File tidak ditemukan!")

# Sistem Kontrol
kepuasan_ctrl = ctrl.ControlSystem(rules)
kepuasan_sim = ctrl.ControlSystemSimulation(kepuasan_ctrl)

# Input Data
kepuasan_sim.input['informasi'] = 80
kepuasan_sim.input['persyaratan'] = 60
kepuasan_sim.input['petugas'] = 50
kepuasan_sim.input['sarpras'] = 90

# Hitung dan Output
try:
    kepuasan_sim.compute()
    hasil = kepuasan_sim.output['kepuasan']
    print(f"Hasil Kepuasan: {hasil:.2f}")

    # 8. Visualisasi
    output_dir = 'visualisasi'
    os.makedirs(output_dir, exist_ok=True)

    kepuasan.view(sim=kepuasan_sim)
    plt.title("Hasil Kepuasan Pelayanan")

    save_path = os.path.join(output_dir, 'hasil.png')
    plt.savefig(save_path)

    print(f"Grafik disimpan di: {save_path}")
    plt.show()

except Exception as e:
    print("Error:", e)