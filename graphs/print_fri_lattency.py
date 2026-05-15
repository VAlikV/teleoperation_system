import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("graphs/Logs_sigma.txt", delimiter=",")

print(data)

t_target_q = data[data[:,0]==0, 1]/1000
target_q = data[data[:,0]==0,2:]

t_current_q = data[data[:,0]==1, 1]/1000
current_q = data[data[:,0]==1,2:]

derivative_target_q = np.gradient(target_q)
derivative_current_q = np.gradient(current_q)


print(target_q.shape)

plt.figure(0)
plt.plot(t_target_q[:], target_q[:,2])
plt.plot(t_current_q[:], current_q[:,2])
# plt.plot([t_target_q[0], t_target_q[-1]], [0,0])
plt.legend(["$q_t$", "$q_c$"])
plt.ylabel("Угол, радианы")
plt.xlabel("Время, мс")

plt.figure(3)
plt.plot(t_target_q[:], derivative_target_q[0][:,2])
plt.plot(t_current_q[:], derivative_current_q[0][:,2])
plt.legend(["$dq_t$", "$dq_c$"])
plt.ylabel("Скорость")
plt.xlabel("Время, мс")
# plt.legend(["Производная Целевое значение угла 3", "Текущее значение угла 3"])
# plt.legend(["t_0", "t_1", "t_2", "c_0", "c_1", "c_2"])


plt.show()


