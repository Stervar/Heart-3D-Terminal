# Импорт необходимых библиотек для создания 3D сердца в терминале
import numpy as np      # Математические операции и работа с массивами
import time             # Работа со временем и задержками
import os               # Работа с операционной системой

def clear_screen():
    """
    Очистка экрана с учетом операционной системы
    
    Использует системную команду cls для Windows и clear для Unix-подобных систем
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def create_heart_points(scale=15):
    """
    Генерация точек для создания 3D модели сердца
    
    Args:
        scale (int): Масштаб сердца
    
    Returns:
        np.array: Массив точек сердца
    """
    # Создание сетки точек с использованием сферических координат
    u = np.linspace(0, 2*np.pi, 100)   # Азимутальный угол
    v = np.linspace(0, np.pi, 50)      # Полярный угол
    
    # Базовые координаты сферы
    x = scale * np.outer(np.cos(u), np.sin(v))
    y = scale * np.outer(np.sin(u), np.sin(v))
    z = scale * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Деформация сферы для создания формы сердца
    x = x * (1 - 0.8 * np.abs(z) / scale)
    y = y * (1 - 0.8 * np.abs(z) / scale) - 0.3 * z
    z = z * 0.8
    
    # Преобразование 2D сетки в массив точек
    return np.column_stack((x.flatten(), y.flatten(), z.flatten()))

def rotate_points(points, angle):
    """
    Вращение точек вокруг оси Y
    
    Args:
        points (np.array): Массив точек
        angle (float): Угол поворота
    
    Returns:
        np.array: Повернутые точки
    """
    # Матрица поворота вокруг оси Y
    rotation_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    
    # Применение матрицы поворота
    return np.dot(points, rotation_matrix.T)

def draw_heart(points, width=80, height=40):
    """
    Отрисовка сердца с использованием блочных символов и z-буфера
    
    Args:
        points (np.array): Точки сердца
        width (int): Ширина экрана
        height (int): Высота экрана
    """
    # Инициализация экрана и z-буфера
    screen = [[' ' for _ in range(width)] for _ in range(height)]
    z_buffer = [[float('-inf') for _ in range(width)] for _ in range(height)]
    
    # Извлечение координат точек
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Масштабирование и центрирование точек
    x = (x / np.max(np.abs(x)) * (width//3)) + width//2
    y = (y / np.max(np.abs(y)) * (height//2)) + height//2
    
    # Блочные символы для создания эффекта затенения
    shading_chars = '░▒▓█'
    
    # Отрисовка точек с учетом глубины
    for xi, yi, zi in zip(x, y, z):
        xi, yi = int(xi), int(yi)
        if 0 <= xi < width and 0 <= yi < height:
            if zi > z_buffer[yi][xi]:
                z_buffer[yi][xi] = zi
                # Расчет интенсивности на основе глубины
                intensity = int((zi - np.min(z)) / (np.max(z) - np.min(z)) * (len(shading_chars) - 1))
                screen[yi][xi] = shading_chars[intensity]
    
    # Вывод экрана
    for row in screen:
        print(''.join(row))

def main():
    """
    Основная функция для запуска анимации сердца
    """
    # Генерация точек сердца
    heart_points = create_heart_points()
    
    # Начальный угол поворота
    angle = 0
    
    try:
        while True:
            # Очистка экрана
            clear_screen()
            
            # Поворот точек
            rotated_points = rotate_points(heart_points, angle)
            
            # Отрисовка сердца
            draw_heart(rotated_points)
            
            # Обновление угла поворота
            angle += 0.1
            
            # Небольшая задержка для управления скоростью анимации
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        # Обработка завершения программы
        print("\nПрограмма завершена")

if __name__ == "__main__":
    try:
        main()  # Запуск основной функции
    except KeyboardInterrupt:
        print("\nПрограмма завершена")