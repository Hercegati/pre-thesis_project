import numpy as np
import matplotlib.pyplot as plt

#### --- Iránykarakterisztika --- ###
def radiation_pattern(theta_deg):
    theta_rad = np.radians(theta_deg)
    gain = np.sin(theta_rad)
    gain = np.maximum(1e-12, gain)
    return gain

### --- ÁLLANDÓK --- ###
lambda_ = 1.0
beta = 2 * np.pi / lambda_

### --- PARAMÉTEREK --- ###
N = 10                  # Antennák száma
d = 0.5                 # Elemközi távolság hullámhosszban (d/lambda)
theta0 = 90            # Fő jel beesési szöge (fok)
theta_i = 100           # Interferencia beesési szöge (fok)
db_minta = 200          # Mintaszám
signal_strength = 1.0        # Főjel abszolút erősség
interference_strength = 0.85  # Interferencia abszolút erősség
noise_sigma =  10           # Zaj szórás
add_noise = True         # Zaj be-/kikapcsolása
add_interference = False     # Interferencia be-/kikapcsolása
add_radiaton_pattern = False   #Iránykaraktersiztika be-/kikapcsolása
resolution = 1801           # 0.1 fokos felbontás


### --- Fő jel generálása --- ###
u = beta * d * np.cos(np.radians(theta0)) #Térbeli frekvencia skalár
Scan_theta0 = np.exp(1j * (np.arange(N)* u)) #Irányvektor 1D tömb
s = np.exp(1j * 2 * np.pi * np.random.rand(db_minta)) #1D tömb
signal_matrix = signal_strength * (Scan_theta0[: , np.newaxis] @ s[np.newaxis , :]) #Mind2 1D tömböl vektort csinálunk így elvégezhető a mátrix szorzás

### --- Interferencia generálása --- ###
u_i = beta * d * np.cos(np.radians(theta_i))
Scan_theta_i = np.exp(1j * (np.arange(N)* u_i))
i_signal = np.exp(1j * 2 * np.pi * np.random.rand(db_minta))
if add_interference:
    interference_matrix = interference_strength * (Scan_theta_i[: , np.newaxis] @ i_signal[np.newaxis , :])
else:
    interference_matrix = np.zeros((N, db_minta), dtype=complex)

### --- Zaj generálása --- ###
if add_noise:
    noise = noise_sigma * (np.random.randn(N, db_minta) + 1j * np.random.randn(N, db_minta)) / np.sqrt(2)
else:
    noise = np.zeros((N, db_minta), dtype=complex)

### --- Mérések összerakása --- ###
r = signal_matrix + interference_matrix + noise

### --- Teljesítmény számítás --- ###
signal_power = np.mean(np.abs(signal_matrix)**2)
noise_power = np.mean(np.abs(noise)**2)
interference_power = np.mean(np.abs(interference_matrix)**2)

### --- SNR --- ###
total_noise_power = noise_power + interference_power

if total_noise_power > 0:
    snr_measured = signal_power / total_noise_power
    snr_measured_dB = 10 * np.log10(snr_measured)
    print(f"Mért SNR (jel / (zaj + interferencia)): {snr_measured_dB:.2f} dB")
else:
    print("Mért SNR: Végtelen (nincs zaj és nincs interferencia)")


### --- Kovariancia mátrix  --- ###
R = (r @ r.conj().T) / db_minta

### --- Fourier teljesítmény spektrum számítása --- ###
angles = np.linspace(0, 180, resolution)
P_fourier = []

for theta in angles:
    u = beta * d * np.cos(np.radians(theta)) #DFT
    if add_radiaton_pattern:
        ant_gain = radiation_pattern(theta) 
        Scan = ant_gain * np.exp(1j * np.arange(N) * u)
    else:
        Scan =np.exp(1j * np.arange(N) * u)
    P = np.abs(Scan.conj().T @ R @ Scan)
    P_fourier.append(P)

### --- Normalizálás --- ###
P_fourier = np.array(P_fourier)
P_fourier /= np.max(P_fourier)

### --- dB skála --- ###
P_db = 10 * np.log10(P_fourier + 1e-12)  # kis offset a log(0) elkerülésére

### --- Shiftelés, hogy minden érték pozitív legyen --- ###
##P_db_shifted = P_db - np.min(P_db)

### --- Ábra megjelenítése --- ###
plt.figure(figsize=(12, 8))
plt.plot(angles, P_db, marker='o', markersize=2, linestyle='-')
plt.xlabel('Theta (fok)')
plt.ylabel('Teljesítmény spektrum (dB)')
plt.title(f'Fourier DOA becslés (N={N}, fő jel beesési szöge={theta0}°, '
          f'Zaj {"BE" if add_noise else "KI"}, Interferencia {"BE" if add_interference else "KI"})')
plt.xlim(0, 180)
plt.grid(True)
plt.show()
