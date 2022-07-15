import numpy as np
import matplotlib.pyplot as plt

b = np.load("loss_list.npy",allow_pickle=True)
print(b)

plt.title('DAGGER')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(b)
plt.show()