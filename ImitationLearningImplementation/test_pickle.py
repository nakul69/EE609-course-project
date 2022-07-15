import pickle
l=[]
for i in range(10):
    l.append(i)

file_name = "sample.pkl"

open_file = open(file_name, "wb")
pickle.dump(l, open_file)
open_file.close()

open_file = open(file_name, "rb")
loaded_list = pickle.load(open_file)
open_file.close()

print(loaded_list)