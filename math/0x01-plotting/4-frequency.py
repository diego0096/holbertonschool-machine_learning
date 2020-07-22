#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

bins = [x * 10 for x in range(11)]
plt.hist(student_grades, align='mid', bins=bins, edgecolor='black')
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.xlim(0, 100)
plt.xticks(bins)
plt.ylim(0, 30)
plt.title('Project A')
plt.show()
