import numpy as np
import sys
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d



dirs = sys.argv[-1]

red = np.load("../data/"+dirs+"/band_1.npy")[1:,1:]
green = np.load("../data/"+dirs+"/band_2.npy")[1:,1:]
blue = np.load("../data/"+dirs+"/band_3.npy")[1:,1:]
nir = np.load("../data/"+dirs+"/band_4.npy")[1:,1:]
ndsm = np.load("../data/"+dirs+"/obj_height.npy")[1:,1:]
slope = np.load("../data/"+dirs+"/slope.npy")[1:,1:]
aspect = np.load("../data/"+dirs+"/aspect.npy")[1:,1:]
dem = np.load("../data/"+dirs+"/dem.npy")[1:,1:]

naip = np.stack([red,green,blue],axis=2)

line = red.shape[0]//2

naip[line,:,0] = np.ones(red.shape[1])*255
naip[line,:,1] = np.ones(red.shape[1])*255
naip[line,:,2] = np.zeros(red.shape[1])

red_line = red[line]
green_line = green[line]
blue_line = blue[line]
nir_line = nir[line]
ndsm_line = ndsm[line]
slope_line = slope[line]
aspect_line = aspect[line]
dem_line = dem[line]


# max_dim = np.amin(np.array([red.shape[0], red.shape[1]]))
# # red = landscape[:max_dim,:max_dim]
# # preds = np.squeeze(preds)
# # cut = preds.shape[0]
# fig = plt.figure(figsize=(8,4))
# ax = fig.add_subplot(111, projection='3d')
# x = np.arange(0,max_dim,1)
# y = np.arange(0,max_dim,1)
# X,Y = np.meshgrid(x,y)
# Z = red[-max_dim:,-max_dim:]
#
# # max = np.amax(max_dim)
# # dem3d=ax.plot_surface(X,Y, Z,cmap='viridis', linewidth=0)
# ax.plot_surface(X,Y, Z,cmap='RdYlGn', linewidth=0)
# # ax1.set_title('Y-NET Prediction '+ str(count))
# # ax.set_zlim3d(0, (max)) #max*3.28084
# plt.show()
# plt.close()





fig, (ax1, ax2, ax3, ax4,ax5,ax6) = plt.subplots(6, 1)
fig.subplots_adjust(hspace=0.5)
ax1.imshow(naip)

ax2.plot(np.arange(red_line.shape[0]), red_line, c='r')
ax2.plot(np.arange(red_line.shape[0]), green_line, c='g')
ax2.plot(np.arange(red_line.shape[0]), blue_line, c='b')
ax2.plot(np.arange(red_line.shape[0]), nir_line, c='c')
ax3.plot(np.arange(red_line.shape[0]), ndsm_line, c='m')
ax4.plot(np.arange(red_line.shape[0]), slope_line)
ax5.plot(np.arange(red_line.shape[0]), aspect_line)
ax6.plot(np.arange(red_line.shape[0]), dem_line)
plt.show()
