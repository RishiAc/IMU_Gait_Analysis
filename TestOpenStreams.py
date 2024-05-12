import pyxdf
import matplotlib.pyplot as plt

# Load the XDF file
streams, fileheader = pyxdf.load_xdf(r"C:\Users\local_user\Downloads\TestForProject\sub-P001_ses-S001_task-Default_run-001_eeg.xdf",synchronize_clocks = True)
timestamps = []
data =  []
# Iterate through streams
for stream in streams:
    if stream['info']['name'][0] == 'your_stream_name':
        data = stream['time_series']
        timestamps = stream['time_stamps']

# Plot the data
plt.plot(timestamps, data)
plt.xlabel('Time (s)')
plt.ylabel('Data')
plt.title('Your Plot Title')
plt.show()