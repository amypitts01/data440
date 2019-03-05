import numpy as np

n = 2000
Sn = 3200 * np.log(81920*n**float(10)+20)
t=True
while(t):
    if(round(Sn,8) == round(n,8)):
        t=False
        print(Sn)
    else:
        n = Sn
        Sn = 3200 * np.log(81920*n**float(10)+20)


