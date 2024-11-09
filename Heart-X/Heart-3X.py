import numpy as np  # Библиотека для работы с многомерными массивами и научных вычислений
import matplotlib.pyplot as plt  # Библиотека для создания статических, анимированных и интерактивных визуализаций
from matplotlib.animation import FuncAnimation  # Класс для создания анимации
import colorsys  # Модуль для преобразования цветовых пространств
import math  # Математические функции

def rotate_points(points, angle_y):
    """
    Функция поворота точек вокруг оси Y
    
    Параметры:
    - points: массив точек для поворота
    - angle_y: угол поворота
    
    Возвращает повернутые точки
    """
    # Создаем матрицу поворота вокруг оси Y
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],     # Первая строка матрицы поворота
        [0, 1, 0],                                 # Вторая строка матрицы (без изменений по Y)
        [-np.sin(angle_y), 0, np.cos(angle_y)]     # Третья строка матрицы поворота
    ])
    # Умножаем точки на транспонированную матрицу поворота
    return np.dot(points, Ry.T)

def create_heart_points(scale=5, num_points=1000, num_layers=30):
    """
    Создание точек для формирования сердца
    
    Параметры:
    - scale: масштаб сердца
    - num_points: количество точек на контуре
    - num_layers: количество слоев для объемности
    
    Возвращает массив точек сердца
    """
    # Создаем параметрический массив для генерации контура сердца
    t = np.linspace(0, 2*np.pi, num_points)
    
    # Параметрические уравнения сердца
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    z = np.zeros_like(x)

    points = []

    # Создание слоев для объемности сердца
    for i in range(num_layers):
        factor = 1 - i/num_layers
        layer_x = factor * x
        layer_y = factor * y
        layer_z = np.full_like(x, -i/2)
        points.extend(zip(layer_x, layer_y, layer_z))

    # Добавление внутренних точек для большей реалистичности
    for _ in range(num_points // 2):
        r = np.random.random() * 0.8
        theta = np.random.random() * 2 * np.pi
        phi = np.random.random() * np.pi
        
        # Вычисление координат внутренних точек с использованием сферических координат
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

# Кэширование базовых точек сердца для оптимизации
BASE_HEART_POINTS = create_heart_points(scale=8)

# Настройка графического окна
fig, ax = plt.subplots(figsize=(10, 8))  # Создание фигуры и осей
plt.style.use('dark_background')  # Темный фон
ax.set_axis_off()  # Убираем оси

# Создание scatter plots для многослойности
scatter_layers = []
for _ in range(10):
    scatter = ax.scatter([], [], c='red', alpha=0.5, s=10)
    scatter_layers.append(scatter)

def animate(frame):
    """
    Функция анимации для каждого кадра
    
    Параметр:
    - frame: номер текущего кадра
    
    Возвращает список обновленных scatter plots
    """
    # Применение эффекта пульсации
    scale = pulsating_effect(frame/10)
    scaled_points = BASE_HEART_POINTS * scale
    
    # Поворот сердца
    rotated_points = rotate_points(scaled_points, frame/10)
    
    # Проекция 3D точек на 2D плоскость
    x = rotated_points[:, 0]
    y = rotated_points[:, 1]
    
    # Динамическое изменение цвета
    hue = (frame/100) % 1.0
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]
    color = (r/255, g/255, b/255)
    
    # Обновление scatter plots для создания объемного эффекта
    updated_scatters = []
    for i, scatter in enumerate(scatter_layers):
        scale_layer = 1 - i * 0.1
        layer_x = x * scale_layer
        layer_y = y * scale_layer
        
        # Обновление параметров scatter plot
        scatter.set_offsets(np.column_stack([layer_x, layer_y]))
        scatter.set_color(color)
        scatter.set_alpha(0.5 - i * 0.05)
        scatter.set_sizes([10-i])
        
        updated_scatters.append(scatter)
    
    # Настройка заголовка и осей
    ax.set_title(f'Animated Heart (Frame: {frame})')
    ax.set_xlim(x.min() * 1.1, x.max() * 1.1)
    ax.set_ylim(y.min() * 1.1, y.max() * 1.1)
    ax.set_aspect('equal')
    
    return updated_scatters

# Создание анимации
anim = FuncAnimation(
    fig,                       # Фигура для анимации
    animate,                   # Функция для каждого кадра
    frames=np.linspace(0, 2*np.pi*10, 200),  # Количество и диапазон кадров
    interval=50,               # Интервал между кадрами (мс)
    blit=True                  # Оптимизация перерисовки
)

plt.tight_layout()  # Автоматическая компоновка
plt.show()  # Отображение анимации