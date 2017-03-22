import subprocess
import sys

sol = 0;
img_index = [200, 400, 500, 1000, 2000, 3000]

for index in img_index:
    print(index)
    for i in range(20):
        print(i)
        path = "./main.out"
        # print(sys.argv[1])
        out = subprocess.Popen([path, "cara" + str(index) + ".jpg"], stdout=subprocess.PIPE)
        output, err = out.communicate()
        list_out = output.split()

        sol += float(list_out[0])
    sol /= 20
    print(sol)
