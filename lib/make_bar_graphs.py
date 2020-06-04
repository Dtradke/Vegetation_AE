
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

plt.figure(figsize=(9, 3))

N = 4

all = (2.58, 6.89, 0.80, 153.79) #LSTM
nodem = (2.64, 6.95, 0.79, 156.28)
noslope = ()




# fft = (0.22, 0.07, 0.07, 0.05)
# all_nofft = (0.13, 0.02, 0.02, 0.02)
# all = (0.18, 0.06, 0.05, 0.05)

fft = np.subtract(fft, norm)
all_nofft = np.subtract(all_nofft, norm)
all = np.subtract(all, norm)


ind = np.arange(N)
width = 0.25
# plt.bar(ind, norm, width, label='Normal')
plt.bar(ind, fft, width, label='FFT-Mode')
plt.bar(ind + (width), all_nofft, width, label='ALL-NoFFT')
plt.bar(ind + (width*2), all, width, label='ALL-Mode')

plt.axhline(y=0, c='black', linewidth=2, label="Normal")
plt.ylabel('F1-Score', fontsize=22, c='black')
# plt.ylim(top=25)
#plt.title('Scores by group and gender')
plt.yticks(fontsize=22, c='black')
plt.xticks(ind + width , ('A1', 'A2', 'A3', 'A4'), fontsize=22, c='black')
plt.legend(loc=1, fontsize=14, ncol=2)
plt.show()
