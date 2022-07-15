import matplotlib.pyplot as plt
import pickle

# file_name = "BCMountainCar.pkl"
# open_file = open(file_name, "rb")
# BcCartPolelist = pickle.load(open_file)
# open_file.close()

file_name2 = "DaggerCartPole.pkl"
open_file = open(file_name2, "rb")
listfinal= pickle.load(open_file)
open_file.close()

plt.figure()
#plt.plot(BcCartPolelist,label="Behvioral Cloning")
plt.plot(listfinal,label="Behavioral Cloning")
plt.plot()
plt.legend()
plt.ylabel("Mean Return")
plt.xlabel("Dagger Iterations")
plt.show()