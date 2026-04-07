# Input Output

Inputs are fed in the input folder, as TOML files listing the properties
of the full system to be simulated. For example, the case of a FMF
system is:

``` toml
# Structured MMF config mirroring uwb_struct.toml layout
[fiber]
dispersion = 1.8e-5
effective_area = 220.71e-12
fiber_length = 70e3

[pulse]
baud_rate = 33e9
pulse_shape = 0

[wdm]
n_modes = 4
n_channels = 200
channel_spacing = 50e9
center_frequency = 195.94e12
launch_power = -5

[amplification]
n_pumps = 6
raman_gain = 0.0

[nlin]
store = true
collision_margin = 5
```

Or, for a case of a UWB SMF system:

``` toml
# general dataset for the UWB
[fiber]
beta2 = 20e-27
effective_area = 80e-12
fiber_length = 70e3
path_to_csv = "input/fiber_data/smf28.csv"

[pulse]
baud_rate = 33e9
pulse_shape = 0

[wdm]
n_modes = 1
center_frequency = 195.94e12
n_channels = 200
channel_spacing = 50e9
launch_power = -5

[wdm.bands.O]
n_channels = 275
launch_power_dbm = 20.0
modulation = "DP-16QAM"
start_nm = 1281.2

[wdm.bands.E]
n_channels = 310
launch_power_dbm = 21.5
modulation = "DP-64QAM"
start_nm = 1340.0

[wdm.bands.S]
n_channels = 296
launch_power_dbm = 20.0
modulation = "DP-64QAM"
start_nm = 1460.0

[wdm.bands.C]
n_channels = 195
launch_power_dbm = 17.0
modulation = "DP-64QAM"
start_nm = 1530.0

[wdm.bands.L]
n_channels = 245
launch_power_dbm = 17.5
modulation = "DP-64QAM"
start_nm = 1565.0

[wdm.bands.U]
n_channels = 119
launch_power_dbm = 18.5
modulation = "DP-64QAM"
start_nm = 1625.0

[amplification]
n_pumps = 6
raman_gain = 0.0

[nlin]
store = true
collision_margin = 5
```
