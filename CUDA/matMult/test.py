import subprocess

sol = 0;
for i in range(20):
    print(i)
    out = subprocess.Popen("./main.out", stdout=subprocess.PIPE)
    output, err = out.communicate()
    sol += float(output)

sol /= 20
print(sol)
