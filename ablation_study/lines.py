import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
plt.figure(figsize=(8,4))

# dem = np.array([0.08, 0, 0.11, -0.14, -0.14, 0.29, -1.38])
# slope = np.array([0.31, 0.89, 0.86, 0.27, 0.04, -0.26, 0.02])
# aspect = np.array([0.15, 0.56, 0.96, 0.36, 0.13, -0.06, -0.04])
#
# ndvi = np.array([0.04, 0.28, 0.29, 0.04, -0.05, 0.52, 1.01])
# red = np.array([0.42, 0.97, 0.88, 0.04, -0.29, -0.86, -0.94])
# green = np.array([0.30, 0.78, 1.04, 0.51, 0.31, -0.32, -0.2])
# blue = np.array([0.2, 0.6, 0.48, -0.03, -0.23, -0.05, 0.96])
# nir = np.array([0.07, 0.33, 0.3, -0.09, -0.24, -0.13, 0.72])
# foot = np.array([0.14, 0.56, 0.61, 0.01, -0.23, -0.26, 0.10])

dem = np.array([0.08, 0.27, 0.74, 0.14, -0.09, -1.26, -1.24])
slope = np.array([0.39, 1.22, 2.04, 0.65, 0.08, -2.13, -2.61])
aspect = np.array([0.36, 1.50, 2.29, 0.61, -0.06, -2.41, -2.29])

ndvi = np.array([1.16, 0.99, 1.56, 0.48, 0.05, -1.75, -2.73])
red = np.array([0.37, 1.19, 1.58, 0.34, -0.17, -1.85, -2.82])
green = np.array([0.31, 0.82, 1.05, 0.16, -0.2, -1.26, 0.95])
blue = np.array([0.15, 0.46, 0.63, 0.26, 0.11, -0.61, -1.7])
nir = np.array([0.25, 1.01, 1.62, 0.52, 0.07, -1.26, -1.88])
foot = np.array([0.28, 1.02, 2.03, 0.76, 0.24, -2.95, -2.94])



lines = [dem, slope, aspect, ndvi, red, green, blue, nir, foot]
x = np.arange(dem.shape[0])
labels = ["!DEM", "!Slope", "!Aspect", "!NDVI", "!Red", "!Green", "!Blue", "!NIR", "!Footprint"]
colors = ['k', 'c', 'y', 'm', 'r', 'g', 'b', 'k', 'c',]
ls = ['-', '-', '-', '--', '--', '--', '--', '--', '--']

for i, val in enumerate(lines):
	plt.plot(x, val, c=colors[i], linestyle=ls[i], label=labels[i])

plt.title("Input Layer Ablation Study", fontsize=20)
plt.ylabel("Error from Y-NET", fontsize=20)
plt.xlabel("Height Classes", fontsize=20)
plt.yticks(fontsize=14)
plt.xticks([0,1,2,3,4,5,6], ["0-2ft", "2-6ft", "6-20ft", "6-50ft", "20-50ft", "50-80ft", "80+ft"], fontsize=14)
plt.legend(ncol=2, loc='best')
plt.show()


#
# xlabels = ["0-2ft", "2-6ft", "6-20ft", "6-50ft", "20-50ft", "50-80ft", "80+ft"]
# xvals = np.arange(len(xlabels))
#
#
# width = 0.075  # the width of the bars
#
# rec = []
# fig, ax = plt.subplots()
# for i, val in enumerate(lines):
# 	if i < 3:
# 		rec.append(ax.bar(xvals - (i*width), lines[i], width, label=labels[i]))
# 	else:
# 		rec.append(ax.bar(xvals + ((i-2)*width), lines[i], width, hatch="x", label=labels[i]))
# # rects2 = ax.bar(x + width/2, women_means, width, label='Women')
# # rects3 = ax.bar(x - width/2, men_means, width, label='Men')
# # rects4 = ax.bar(x + width/2, women_means, width, label='Women')
# # rects5 = ax.bar(x - width/2, men_means, width, label='Men')
# # rects6 = ax.bar(x + width/2, women_means, width, label='Women')
# # rects7 = ax.bar(x - width/2, men_means, width, label='Men')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel("Error from Y-NET", fontsize=20)
# ax.set_xlabel("Height Classes", fontsize=20)
# ax.set_title("Input Layer Ablation Study", fontsize=20)
# ax.set_xticks(xvals)
# ax.set_xticklabels(xlabels, fontsize=15)
# ax.legend()
#
#
# # def autolabel(rects):
# #     """Attach a text label above each bar in *rects*, displaying its height."""
# #     for rect in rects:
# #         height = rect.get_height()
# #         ax.annotate('{}'.format(height),
# #                     xy=(rect.get_x() + rect.get_width() / 7, height),
# #                     xytext=(0, 3),  # 3 points vertical offset
# #                     textcoords="offset points",
# #                     ha='center', va='bottom')
# #
# # for i in rec:
# # 	autolabel(i)
# # # autolabel(rects2)
#
# fig.tight_layout()
#
# plt.show()
