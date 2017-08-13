import matplotlib.pyplot as plt
import matplotlib.animation as animation

class LivePlot:
    """ Live Plot 
    
    Parameters
    ----------
    ticker_range: Int
        Integer range x axis plots points
    labels: Tuple
        Strings of x axis label and y axis label

    Returns
    ------
    Scatterplot

    """ 

    def __init__(self,ticker_range=1000,labels=["Title","x","y"]):
        self.labels = labels 
        self.ticker_range = ticker_range
        self.plt = plt
        self.fig = self.plt.figure(1)
        self.ax = self.fig.add_subplot(1,1,1)
        self.X = []
        self.Y = []
    def set_properties(self):
        self.plt.title(self.labels[0])
        self.plt.xlabel(self.labels[1])
        self.plt.ylabel(self.labels[2])
        self.plt.set_ylim = (0,1)

    def add_point(self,x_t,y_t):
        """ Add new point to plot"""
        self.X.append(x_t)
        self.Y.append(y_t)
    def animate(self)
        self.plt.plot(self.X,self.Y)
    def make_animation(self):
        a = animation.FuncAnimation(self.fig,animate,interval=1000)
        return a,self.plot


