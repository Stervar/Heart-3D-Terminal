# Импорт необходимых библиотек
import numpy as np  # Библиотека для численных вычислений и работы с многомерными массивами
import matplotlib.pyplot as plt  # Библиотека для создания статических, анимированных и интерактивных визуализаций
from matplotlib.animation import FuncAnimation  # Класс для создания анимации
import colorsys  # Модуль для преобразования цветовых пространств (HSV в RGB)
import math  # Математические функции (синус, косинус и др.)

def rotate_points(points, angle_y):
    """
    Функция поворота точек вокруг оси Y в трехмерном пространстве
    
    Параметры:
    - points: массив трехмерных точек для поворота
    - angle_y: угол поворота вокруг оси Y в радианах
    
    Возвращает массив повернутых точек
    """
    # Создаем матрицу поворота вокруг оси Y
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],     # Первая строка матрицы поворота
        [0, 1, 0],                                 # Вторая строка (ось Y не меняется)
        [-np.sin(angle_y), 0, np.cos(angle_y)]     # Третья строка матрицы поворота
    ])
    
    # Умножаем точки на транспонированную матрицу поворота
    return np.dot(points, Ry.T)

def create_heart_points(scale=5, num_points=1000, num_layers=30):
    """
    Создание точек для формирования 3D-сердца
    
    Параметры:
    - scale: масштаб сердца
    - num_points: количество точек на контуре
    - num_layers: количество слоев для объемности
    
    Возвращает массив точек сердца
    """
    # Создаем параметрический массив для генерации контура сердца
    t = np.linspace(0, 2*np.pi, num_points)
    
    # Параметрические уравнения сердца (математическая формула контура)
    x = 16 * np.sin(t)**3  # X-координата
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)  # Y-координата
    z = np.zeros_like(x)  # Z-координата (изначально нулевая)

    points = []

    # Создание слоев для объемности сердца
    for i in range(num_layers):
        factor = 1 - i/num_layers  # Коэффициент уменьшения размера слоя
        layer_x = factor * x  # X-координаты слоя
        layer_y = factor * y  # Y-координаты слоя
        layer_z = np.full_like(x, -i/2)  # Z-координаты слоя (смещение вглубь)
        points.extend(zip(layer_x, layer_y, layer_z))  # Добавление точек слоя

    # Добавление внутренних точек для большей реалистичности
    for _ in range(num_points // 2):
        # Случайные сферические координаты
        r = np.random.random() * 0.8  # Радиус
        theta = np.random.random() * 2 * np.pi  # Азимутальный угол
        phi = np.random.random() * np.pi  # Полярный угол
        
        # Вычисление декартовых координат
        x = r * 16 * np.sin(theta)**3 * np.sin(phi)
        y = r * (13 * np.cos(theta) - 5 * np.cos(2*theta) - 
                2 * np.cos(3*theta) - np.cos(4*theta)) * np.sin(phi)
        z = r * 15 * np.cos(phi)
        points.append((x, y, z))

    # Масштабирование и преобразование в numpy-массив
    return scale * np.array(points)

def pulsating_effect(time):
    """
    Создание эффекта пульсации с использованием синусоиды
    
    Параметр:
    - time: текущее время для анимации
    
    Возвращает коэффициент масштабирования
    """
    return 1 + 0.05 * math.sin(time * 2)

def animate(frame):
    """
    Функция анимации для каждого кадра
    
    Параметр:
    - frame: номер текущего кадра
    """
    # Очистка текущего графика
    plt.clf()
    
    # Создание начальных точек сердца
    heart_points = create_heart_points(scale=8)
    
    # Применение эффекта пульсации
    scale = pulsating_effect(frame/10)
    scaled_points = heart_points * scale
    
    # Поворот сердца вокруг оси Y
    rotated_points = rotate_points(scaled_points, frame/10)
    
    # Извлечение координат
    x = rotated_points[:, 0]  # X-координаты
    y = rotated_points[:, 1]  # Y-координаты
    z = rotated_points[:, 2]  # Z-координаты
    
    # Динамическое изменение цвета
    hue = (frame/100) % 1.0  # Циклическое изменение оттенка
    # Преобразование HSV в RGB
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]
    color = (r/255, g/255, b/255)  # Нормализация цвета
    
    # Создание объемного эффекта через многослойный scatter
    for i in range(10):
        scale_layer = 1 - i * 0.1  # Уменьшение размера слоя
        plt.scatter(
            x * scale_layer,  # X-координаты слоя
            y * scale_layer,  # Y-координаты слоя
            c=[color],  # Цвет точек
            alpha=0.5 - i * 0.05,  # Прозрачность слоя
            s=10-i  # Размер точек
        )
    
    # Настройка заголовка и параметров графика
    plt.title(f'Animated Heart (Frame: {frame})')
    plt.axis('equal')  # Равномерный масштаб осей
    plt.grid(True, linestyle='--', alpha=0.3)  # Сетка графика

# Настройка графического окна
plt.figure(figsize=(10, 8))  # Размер окна
plt.style.use('dark_background')  # Темный стиль

# Создание анимации
anim = FuncAnimation(
    plt.gcf(),  # Текущая фигура
    animate,  # Функция анимации
    frames=np.linspace(0, 2*np.pi*10, 200),  # Кад ровое пространство для анимации
    interval=50  # Интервал между кадрами в миллисекундах
)

plt.show()  # Отображение анимации на экране