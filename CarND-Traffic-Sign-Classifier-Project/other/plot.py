import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

X = [590,540,740,130,810,300,320,230,470,620,770,250]
Y = [32,36,39,52,61,72,77,75,68,57,48,48]

plt.scatter(X,Y)
plt.title('Relationship Between Temperature and Iced Coffee Sales')
plt.xlabel('Cups of Iced Coffee Sold')
plt.ylabel('Temperature in Fahrenheit')
plt.show()
