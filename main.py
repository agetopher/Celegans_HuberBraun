import settings
from Fit_HuberBraun_Matrix_5param_AVB_First import Fit_HuberBraun_Matrix_5param_AVB_First

settings.init()

params = [44, 0.000363000000000000, 5.45726102941647e-06, 0.00167208415554239,0.000660000000000000]
r1 = params[0]
ubestAVB = params[1:]

t, u = Fit_HuberBraun_Matrix_5param_AVB_First(ubestAVB, r1)

print("Done")