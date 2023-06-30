import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
data = io.loadmat('./results/thai_statue/wire.mat')
best_img = data['best_img'].astype(np.float32)
img = data['img'].astype(np.float32)

# img = img.reshape(*img.shape,1)

[XS,YS,ZS,L] = best_img.shape

x = np.linspace(-1,1,XS)
y = np.linspace(-1,1,YS)
z = np.linspace(-1,1,ZS)
colors = np.reshape(best_img,[XS*YS*ZS,L])
inds = np.sum(colors, axis = 1) > 3
colors = colors[inds,:]
colors = colors / np.max(colors)
colors[colors < 0] = 0

[X,Y,Z] = np.meshgrid(x,y,z,indexing='ij')
points = np.stack((X,Y,Z), axis = 3)
points = np.reshape(points, [XS*YS*ZS, 3])
points = points[inds,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], c=colors[:,[26,16,6]], marker='o')
# ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')
# ax.set_xlim(-0.5,0.5)
# ax.set_ylim(-0.5,0.5)
# ax.set_zlim(0,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

a = 1