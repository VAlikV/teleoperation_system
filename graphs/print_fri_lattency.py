import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("graphs/Logs.txt", delimiter=",")

print(data)

t_target_q = data[data[:,0]==0, 1]
target_q = data[data[:,0]==0,2:]

t_current_q = data[data[:,0]==1, 1]
current_q = data[data[:,0]==1,2:]

derivative_target_q = np.gradient(target_q)
derivative_current_q = np.gradient(current_q)


print(target_q.shape)

plt.figure(0)
plt.plot(t_target_q[:], target_q[:,1])
plt.plot(t_current_q[:], current_q[:,1])
plt.plot([t_target_q[0], t_target_q[-1]], [0,0])
plt.legend(["target position joint 0", "current position joint 0"])

plt.figure(3)
plt.plot(t_target_q[:], derivative_target_q[0][:,0])
plt.plot(t_current_q[:], derivative_current_q[0][:,0])
# plt.legend(["t_0", "t_1", "t_2", "c_0", "c_1", "c_2"])


plt.show()


