from MLPlot import LivePlot
import matplotlib.pyplot as plt
import numpy as np
from time import sleep


XS = np.arange(20)
YS = np.power(XS,2)
lp = LivePlot(ticker_range=1)
lp.set_properties()
for x,y in zip(XS,YS):
    sleep(1)
    lp.add_point(x,y)
    
