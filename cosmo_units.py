# cosmo_units.py

# --- Physical Constants ---
c = 2.99792458e8  # speed of light in m/s
G = 6.67430e-11   # gravitational constant in m^3 kg^-1 s^-2
h_P = 6.62607015e-34  # Planck constant in J*s
k_B = 1.380649e-23  # Boltzmann constant in J/K
sigma_T = 6.6524587158e-29  # Thomson cross-section in m^2
m_p = 1.67262192369e-27  # proton mass in kg
m_e = 9.1093837015e-31  # electron mass in kg
e = 1.602176634e-19  # elementary charge in C
H0 = 70.0  # Hubble constant in km/s/Mpc (can be updated based on your model)

# --- Unit Conversions ---
Mpc_to_m = 3.085677581e22  # 1 Mpc in meters
pc_to_m = 3.085677581e16   # 1 parsec in meters
ly_to_m = 9.4607e15        # 1 light-year in meters
km_to_m = 1e3              # 1 km in meters
eV_to_J = 1.602176634e-19  # 1 eV in Joules
J_to_eV = 1 / eV_to_J
K_to_eV = k_B / eV_to_J    # temperature in Kelvin to eV

# --- Derived Quantities ---
# Hubble time and distance (in SI)
H0_SI = H0 * 1000 / Mpc_to_m  # Convert H0 to s^-1
t_Hubble = 1 / H0_SI  # Hubble time in seconds
d_Hubble = c * t_Hubble  # Hubble distance in meters

# --- Cosmology Helper Functions (optional) ---
def hubble_parameter(z, Omega_m=0.3, Omega_L=0.7, H0_kmsmpc=H0):
    """Calculate H(z) in km/s/Mpc assuming flat ΛCDM."""
    return H0_kmsmpc * ((Omega_m * (1 + z)**3 + Omega_L)**0.5)

def age_of_universe(H0_kmsmpc=H0):
    """Return Hubble time in Gyr."""
    seconds_in_Gyr = 3.1536e16
    return t_Hubble / seconds_in_Gyr

# --- Optional: Group in dictionaries for quick access ---
constants = {
    "c": c,
    "G": G,
    "h_P": h_P,
    "k_B": k_B,
    "sigma_T": sigma_T,
    "m_p": m_p,
    "m_e": m_e,
    "e": e,
    "H0": H0,
}

conversions = {
    "Mpc_to_m": Mpc_to_m,
    "pc_to_m": pc_to_m,
    "ly_to_m": ly_to_m,
    "eV_to_J": eV_to_J,
    "J_to_eV": J_to_eV,
    "K_to_eV": K_to_eV,
}
