import subprocess
import sys
import matplotlib.pyplot as plt

sol = 0
img_index = [200, 400, 500, 1000, 2000, 3000, 5000, 10000, 20000]
binary = ["seq", "seqOCV", "par", "parOCV"]

for exe in binary:
    x = []
    y = []

    for index in img_index:
        # print(index)
        for i in range(20):
            # print(i)
            path = "./" + exe + ".out"
            # print(sys.argv[1])
            out = subprocess.Popen(
                [path, "cara" + str(index) + ".jpg"], stdout=subprocess.PIPE)
            output, err = out.communicate()
            # print output
            # print err
            list_out = output.split()

            sol += float(list_out[0])
        sol /= 20
        print(sol)
        x.append(index)
        y.append(sol)

# plt.plot(x, y, 'r')
# plt.show()
