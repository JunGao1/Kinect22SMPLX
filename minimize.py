import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Uncons_Minim(object):
    
    def objective(self, x, coeffs):
        return coeffs[0] * x ** 2 + coeffs[1] * x + coeffs[2]

    def exec(self):
        x0 = 3.0
        mycoeffs = [1.0 , -2.0 , 0.0]
        myoptions = {'disp' : True}
        results = opt.minimize(self.objective, x0=x0, args=mycoeffs, options=myoptions)

        print("Solution: x = %f" % results.x)
        print(f"Solution: x = {results.x}")
        print(results)

class Global_Minim(object):
    
    def eggholder(self, x):
        return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
                -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))
    
    def draw_3D(self):
        x = np.arange(-512, 513)
        y = np.arange(-512, 513)
        xgrid, ygrid = np.meshgrid(x, y)
        xy = np.stack([xgrid, ygrid])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(45, -45)
        ax.plot_surface(xgrid, ygrid, self.eggholder(xy), cmap='terrain')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('eggholder(x, y)')
        plt.show()

if __name__ == '__main__':
    
    #Uncons_Minim().exec()
    Global_Minim().draw_3D()