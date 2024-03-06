import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
import time
import random
from matplotlib.animation import PillowWriter


#Генератор списков
def list_generate(n):
    lists = [random.randint(0,100) for i in range(n)]
    return lists

#счет времени numpy
def time_numpy_counter(array1, array2):
    time_start_numpy = time.perf_counter()
    array_for_counter = numpy.multiply(array1, array2)
    time_end_numpy = time.perf_counter()
    return time_end_numpy - time_start_numpy

#счет времени list
def time_list_counter(array1, array2):
    time_start_list = time.perf_counter()
    c = [a * b for a, b in zip(array1, array2)]
    time_end_list = time.perf_counter()
    return time_end_list - time_start_list

#сравнение 
def task1_counters():
    list_first = list_generate(1000000)
    list_second = list_generate(1000000)
    print('Время выполнения со списками: ', time_list_counter(list_first, list_second))
    array_numpy_first = np.random.randint(1000, size=1_000_000)
    array_numpy_second = np.random.randint(1000, size=1_000_000)
    print('Время выполнения с numpy: ', time_numpy_counter(array_numpy_first, array_numpy_second))
    print('Numpy быстрее списка в ', float(time_list_counter(list_first, list_second)) /
          float(time_numpy_counter(array_numpy_first, array_numpy_second)),
          ' раз')

#2 задание
def task2_histogram():
    ar = np.genfromtxt('data2.csv', delimiter=',')
    ar = ar[1:]

    ph = np.array(ar[:, 0], float)
    ph = ph[~np.isnan(ph)]

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot()

    axis.hist(ph, 50, (0, 20), color='lightblue', ec='blue')
    axis.grid()

    plt.title('гистограмма')
    plt.xlabel(f'значения\nсреднеквадратичное отклонение: {np.std(ph): 0.2f}')
    plt.ylabel('частота')

    plt.show()


def task3_schedule():
    x = np.linspace(-2*np.pi, 2*np.pi, 20)
    y = np.sin(x)*np.cos(x)
    z = np.sin(x)*np.cos(x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, marker='o', c='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D')
    plt.show()

def upd_task_animation():
    figure = plt.figure()
    l, = plt.plot([], [], 'k')

    plt.xlim(-10, 10)
    plt.ylim(-3, 3)

    write = PillowWriter(fps=30)

    x_values = []
    y_values = []

    with write.saving(figure, "y=sin(x).gif", 100):
        for x in np.linspace(-10, 10, 100):
            x_values.append(x)
            y_values.append(np.sin(x))

            l.set_data(x_values, y_values)
            write.grab_frame()

if __name__ == '__main__':
    task1_counters() #1

    task2_histogram() #2

    task3_schedule() #3

    upd_task_animation() #дополнительное