import open3d as o3
import numpy as np
import json
import torch

#class_mapping = {'Bicycle': 1, 'VanSUV': 2, 'UtilityVehicle': 3, 'Cyclist': 4, 'Truck': 5,
#                  'Car': 6, 'EmergencyVehicle': 7, 'MotorBiker': 8, 'Motorcycle': 9,
#                    'CaravanTransporter': 10, 'Animal': 11, 'Bus': 12, 'Pedestrian': 13,
#                      'Trailer': 14}

new_class_mapping = {
    'TwoWheeled': 1,     # Bicycle, Motorcycle
    'Rider': 2,          # Cyclist, MotorBiker
    'Car': 3,            # Car, VanSUV, UtilityVehicle, EmergencyVehicle
    'LargeVehicle': 4,   # Truck, Bus
    'Trailer': 5,        # CaravanTransporter, Trailer
    'Pedestrian': 6      # Pedestrian, Animal
}

class_mapping = {'Bicycle': 1, 'VanSUV': 3, 'UtilityVehicle': 3, 'Cyclist': 2, 'Truck': 4,
                  'Car': 3, 'EmergencyVehicle': 3, 'MotorBiker': 2, 'Motorcycle': 1,
                    'CaravanTransporter': 5, 'Animal': 6, 'Bus': 4, 'Pedestrian': 6,
                      'Trailer': 5}

# Create array of RGB colour values from the given array of reflectance values
def colours_from_reflectances(reflectances):
    return np.stack([reflectances, reflectances, reflectances], axis=1)

def create_open3d_pc(lidar, cam_image=None):
    # create open3d point cloud
    pcd = o3.geometry.PointCloud()
    
    # assign point coordinates
    pcd.points = o3.utility.Vector3dVector(lidar['points'])
    
    # assign colours
    if cam_image is None:
        median_reflectance = np.median(lidar['reflectance'])
        colours = colours_from_reflectances(lidar['reflectance']) / (median_reflectance * 5)
        
        # clip colours for visualisation on a white background
        colours = np.clip(colours, 0, 0.75)
    else:
        rows = (lidar['row'] + 0.5).astype(np.int)
        cols = (lidar['col'] + 0.5).astype(np.int)
        colours = cam_image[rows, cols, :] / 255.0
        
    pcd.colors = o3.utility.Vector3dVector(colours)
    
    return pcd

def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('\\')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + \
                        'camera_' + \
                        file_name_image[2] + '_' + \
                        file_name_image[3] + '.png'

    return file_name_image

def extract_semantic_file_name_from_image_file_name(file_name_image):
    file_name_semantic_label = file_name_image.split('\\')
    file_name_semantic_label = file_name_semantic_label[-1].split('.')[0]
    file_name_semantic_label = file_name_semantic_label.split('_')
    file_name_semantic_label = file_name_semantic_label[0] + '_' + \
                  'label_' + \
                  file_name_semantic_label[2] + '_' + \
                  file_name_semantic_label[3] + '.png'
    
    return file_name_semantic_label


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q
    
def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
    image = np.copy(image_orig)
    
    # get rows and cols
    rows = (lidar['row'] + 0.5).astype(int)
    cols = (lidar['col'] + 0.5).astype(int)
  
    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar['distance'])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar['distance'])

    # get distances
    distances = lidar['distance']  
    # determine point colours from distance
    colours = (distances - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, \
                        np.sqrt(pixel_opacity), 1.0)) for c in colours])
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols, :] = \
                (1. - pixel_opacity) * \
                np.multiply(image[pixel_rows, pixel_cols, :], \
                colours[i]) + pixel_opacity * 255 * colours[i]
    return image.astype(np.uint8)

def skew_sym_matrix(u):
    return np.array([[    0, -u[2],  u[1]], 
                     [ u[2],     0, -u[0]], 
                     [-u[1],  u[0],    0]])

def axis_angle_to_rotation_mat(axis, angle):
    return np.cos(angle) * np.eye(3) + \
        np.sin(angle) * skew_sym_matrix(axis) + \
        (1 - np.cos(angle)) * np.outer(axis, axis)

def read_bounding_boxes(file_name_bboxes):
    # open the file
    with open (file_name_bboxes, 'r') as f:
        bboxes = json.load(f)
        
    boxes = [] # a list for containing bounding boxes  
    #print(bboxes.keys())
    
    for bbox in bboxes.keys():
        bbox_read = {} # a dictionary for a given bounding box
        bbox_read['class'] = bboxes[bbox]['class']
        bbox_read['truncation']= bboxes[bbox]['truncation']
        bbox_read['occlusion']= bboxes[bbox]['occlusion']
        bbox_read['alpha']= bboxes[bbox]['alpha']
        bbox_read['top'] = bboxes[bbox]['2d_bbox'][0]
        bbox_read['left'] = bboxes[bbox]['2d_bbox'][1]
        bbox_read['bottom'] = bboxes[bbox]['2d_bbox'][2]
        bbox_read['right']= bboxes[bbox]['2d_bbox'][3]
        bbox_read['center'] =  np.array(bboxes[bbox]['center'])
        bbox_read['size'] =  np.array(bboxes[bbox]['size'])
        angle = bboxes[bbox]['rot_angle']
        axis = np.array(bboxes[bbox]['axis'])
        bbox_read['rotation'] = axis_angle_to_rotation_mat(axis, angle) 
        boxes.append(bbox_read)

    return boxes

def read_bounding_boxes_2d(file_name_bboxes, image_size, output_format='DETR'):
    """
    Загружает 2D боксы и конвертирует в указанный формат.
    
    Аргументы:
        file_name_bboxes (str): путь к файлу с боксами (JSON)
        image_size (tuple): (image_width, image_height)
        output_format (str): 'SSD', 'YOLO' или 'DETR'
    
    Возвращает:
        dict: {
            "labels": Tensor(N,),
            "boxes": Tensor(N, 4) или список словарей (если SSD)
        }
    """
    image_width, image_height = image_size

    with open(file_name_bboxes, 'r') as f:
        bboxes = json.load(f)

    labels = []
    boxes = []

    for key, obj in bboxes.items():
        cls_id = class_mapping[obj['class']]
        labels.append(cls_id)

        # Исходные абсолютные координаты
        top, left, bottom, right = obj['2d_bbox']

        if output_format.upper() == 'SSD':
            #boxes.append([left, top, right, bottom])
            boxes.append([top, left, bottom, right])


        elif output_format.upper() in ['YOLO', 'DETR']:
            # Конвертация в формат [cx, cy, w, h] (нормализованные)
            x_center = (left + right) / 2.0 / image_width
            y_center = (top + bottom) / 2.0 / image_height
            width = (right - left) / image_width
            height = (bottom - top) / image_height
            boxes.append([x_center, y_center, width, height])

        else:
            raise ValueError(f"Unknown format: {output_format}. Use 'SSD', 'YOLO' or 'DETR'")

    return {
        "labels": torch.tensor(labels, dtype=torch.long),
        "boxes": torch.tensor(boxes, dtype=torch.float32)
    }

# Матрица преобразования из системы лидара в систему камеры
T_lidar_to_camera = np.array([
    [0, -1, 0],  # Xкам = -Yлид
    [0, 0, -1],  # Yкам = -Zлид
    [1, 0, 0]    # Zкам = Xлид
])

def read_bounding_boxes_3d(file_name_bboxes, image_size, output_format='DETR'):
    """
    Загружает 3D боксы и конвертирует в указанный формат.
    
    Аргументы:
        file_name_bboxes (str): путь к файлу с боксами (JSON)
        image_size (tuple): (image_width, image_height)
        output_format (str): 'SSD', 'YOLO' или 'DETR'
    
    Возвращает:
        dict: {
            "labels": Tensor(N,),
            "boxes": Tensor(N, 7) или список словарей (если SSD)
        }
    """
    image_width, image_height = image_size

    with open(file_name_bboxes, 'r') as f:
        bboxes = json.load(f)

    labels = []
    boxes = []

    for key, obj in bboxes.items():
        cls_id = class_mapping[obj['class']]
        labels.append(cls_id)

        # Центр в системе камеры
        center = np.dot(obj['center'], T_lidar_to_camera.T)
        x, y, z = center
        dx, dy, dz = obj['size']
        yaw = obj['rot_angle']

        if output_format.upper() == 'SSD':
            boxes.append([x, y, z, dx, dy, dz, yaw])

        elif output_format.upper() in ['YOLO', 'DETR']:
            # Нормализация X,Y и размеров по ширине/высоте изображения
            x_norm = x / image_width
            y_norm = y / image_height
            dx_norm = dx / image_width
            dy_norm = dy / image_height
            # Z и высоту оставляем в абсолютных значениях
            boxes.append([x_norm, y_norm, z, dx_norm, dy_norm, dz, yaw])
        else:
            raise ValueError(f"Unknown format: {output_format}. Use 'SSD', 'YOLO' or 'DETR'")

    return {
        "labels": torch.tensor(labels, dtype=torch.long),
        "boxes": torch.tensor(boxes, dtype=torch.float32)
    }

def extract_bboxes_file_name_from_image_file_name(file_name_image):
    file_name_bboxes = file_name_image.split('\\')
    file_name_bboxes = file_name_bboxes[-1].split('.')[0]
    file_name_bboxes = file_name_bboxes.split('_')
    file_name_bboxes = file_name_bboxes[0] + '_' + \
                  'label3D_' + \
                  file_name_bboxes[2] + '_' + \
                  file_name_bboxes[3] + '.json'
    
    return file_name_bboxes

def get_points(bbox):
    half_size = bbox['size'] / 2.
    
    if half_size[0] > 0:
        # calculate unrotated corner point offsets relative to center
        brl = np.asarray([-half_size[0], +half_size[1], -half_size[2]])
        bfl = np.asarray([+half_size[0], +half_size[1], -half_size[2]])
        bfr = np.asarray([+half_size[0], -half_size[1], -half_size[2]])
        brr = np.asarray([-half_size[0], -half_size[1], -half_size[2]])
        trl = np.asarray([-half_size[0], +half_size[1], +half_size[2]])
        tfl = np.asarray([+half_size[0], +half_size[1], +half_size[2]])
        tfr = np.asarray([+half_size[0], -half_size[1], +half_size[2]])
        trr = np.asarray([-half_size[0], -half_size[1], +half_size[2]])
     
        # rotate points
        points = np.asarray([brl, bfl, bfr, brr, trl, tfl, tfr, trr])
        points = np.dot(points, bbox['rotation'].T)
        
        # add center position
        points = points + bbox['center']
  
    return points

# Create or update open3d wire frame geometry for the given bounding boxes
def _get_bboxes_wire_frames(bboxes, linesets=None, color=None):

    num_boxes = len(bboxes)
        
    # initialize linesets, if not given
    if linesets is None:
        linesets = [o3.geometry.LineSet() for _ in range(num_boxes)]

    # set default color
    if color is None:
        #color = [1, 0, 0]
        color = [0, 0, 1]

    assert len(linesets) == num_boxes, "Number of linesets must equal number of bounding boxes"

    # point indices defining bounding box edges
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [0, 4], [1, 5], [2, 6], [3, 7],
             [4, 5], [5, 6], [6, 7], [7, 4], 
             [5, 2], [1, 6]]

    # loop over all bounding boxes
    for i in range(num_boxes):
        # get bounding box corner points
        points = get_points(bboxes[i])
        # update corresponding Open3d line set
        colors = [color for _ in range(len(lines))]
        line_set = linesets[i]
        line_set.points = o3.utility.Vector3dVector(points)
        line_set.lines = o3.utility.Vector2iVector(lines)
        line_set.colors = o3.utility.Vector3dVector(colors)

    return linesets

# Матрица преобразования из системы лидара в систему камеры
T_lidar_to_camera = np.array([
    [0, -1, 0],  # Xкам = -Yлид
    [0, 0, -1],  # Yкам = -Zлид
    [1, 0, 0]    # Zкам = Xлид
])

def transform_lidar_to_camera(points):
    # Преобразование облака точек
    points_camera = np.dot(points, T_lidar_to_camera.T)

    return points_camera


def transform_boxes_to_camera(bounding_boxes_data):
    # Преобразование боксов
    boxes_camera = []
    for bbox in bounding_boxes_data:
        # Преобразование центра
        center_camera = np.dot(bbox['center'], T_lidar_to_camera.T)
        
        # Преобразование матрицы поворота
        # Новая ориентация = T_lidar_to_camera * R_lidar
        rotation_camera = np.dot(T_lidar_to_camera, bbox['rotation'])
        
        # Размеры остаются неизменными, так как это локальные параметры бокса
        size_camera = bbox['size'].copy()
        
        # Создаем новый словарь с преобразованными данными
        bbox_camera = {
            'center': center_camera,
            'size': size_camera,
            'rotation': rotation_camera,
            'class': bbox['class'],
            'top': bbox['top'],
            'left': bbox['left'],
            'bottom': bbox['bottom'],
            'right': bbox['right']
        }
        boxes_camera.append(bbox_camera)
    
    return boxes_camera


def transform_lidar_and_boxes_to_camera(points, bounding_boxes_data):
    """
    Преобразует облако точек и параметры боксов из системы лидара в систему камеры.
    
    Args:
        points (np.ndarray): Облако точек лидара размером (N, 3) в системе лидара.
        bounding_boxes_data (list): Список словарей с параметрами боксов:
            {'center': np.ndarray (3,), 'size': np.ndarray (3,), 'rotation': np.ndarray (3,3), 'class': str}
    
    Returns:
        tuple: (points_camera, boxes_camera)
            - points_camera (np.ndarray): Облако точек в системе камеры (N, 3).
            - boxes_camera (list): Список словарей с преобразованными параметрами боксов.
    """
    # Преобразование облака точек
    points_camera = np.dot(points, T_lidar_to_camera.T)
    
    # Преобразование боксов
    boxes_camera = []
    for bbox in bounding_boxes_data:
        # Преобразование центра
        center_camera = np.dot(bbox['center'], T_lidar_to_camera.T)
        
        # Преобразование матрицы поворота
        # Новая ориентация = T_lidar_to_camera * R_lidar
        rotation_camera = np.dot(T_lidar_to_camera, bbox['rotation'])
        
        # Размеры остаются неизменными, так как это локальные параметры бокса
        size_camera = bbox['size'].copy()
        
        # Создаем новый словарь с преобразованными данными
        bbox_camera = {
            'center': center_camera,
            'size': size_camera,
            'rotation': rotation_camera,
            'class': bbox['class']
        }
        boxes_camera.append(bbox_camera)
    
    return points_camera, boxes_camera


def yaw_to_rotation(yaw):
    cos_theta = np.cos(yaw)
    sin_theta = np.sin(yaw)
    return np.array([
        [cos_theta, -sin_theta, 0],
        [0, 0, -1],
        [sin_theta, cos_theta, 0]
    ])

# Внутренние параметры камеры
K = np.array([
    [1.68733691e+03, 0.00000000e+00, 9.65434141e+02],
    [0.00000000e+00, 1.78342847e+03, 6.84419360e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
image_width = 1920
image_height = 1208

# Функция для получения 3D-углов бокса
def get_3d_corners(box_3d):

    x, y, z, dx, dy, dz, yaw = box_3d
    center = np.array([x, y, z])
    size = np.array([dx, dy, dz])  # [длина, ширина, высота]
    
    L, W, H = size
    corners = np.array([
        [-L/2, -W/2, -H/2], [ L/2, -W/2, -H/2], [ L/2,  W/2, -H/2], [-L/2,  W/2, -H/2],  # Нижняя плоскость
        [-L/2, -W/2,  H/2], [ L/2, -W/2,  H/2], [ L/2,  W/2,  H/2], [-L/2,  W/2,  H/2]   # Верхняя плоскость
    ])
    
    rotation = yaw_to_rotation(yaw)
    corners_rotated = np.dot(corners, rotation.T)
    return corners_rotated + center

# Функция проекции 3D в 2D
def project_3d_to_2d(points_3d, K):
    points_2d = np.dot(K, points_3d.T).T
    points_2d[:, 0] /= points_2d[:, 2]  # u = x/z
    points_2d[:, 1] /= points_2d[:, 2]  # v = y/z
    return points_2d[:, :2]

# Функция для вычисления 2D-баундинг-бокса
def get_2d_bounding_box(corners_2d):
    """
    Вычисляет 2D-баундинг-бокс в формате [x_min, y_min, x_max, y_max].
    
    Args:
        corners_2d (np.ndarray): 2D-координаты углов (N, 2).
    
    Returns:
        list: [x_min, y_min, x_max, y_max].
    """
    x_min = np.min(corners_2d[:, 0])
    y_min = np.min(corners_2d[:, 1])
    x_max = np.max(corners_2d[:, 0])
    y_max = np.max(corners_2d[:, 1])
    return [x_min, y_min, x_max, y_max]

def get_2d_bboxes_from_3d(boxes_3d):

    bboxes_2d = []

    for i, box_3d in enumerate(boxes_3d):
        # Получаем 3D-углы
        corners_3d = get_3d_corners(box_3d)
        
        # Проецируем в 2D
        corners_2d = project_3d_to_2d(corners_3d, K)
        
        # Вычисляем 2D-баундинг-бокс
        bbox_2d = get_2d_bounding_box(corners_2d)
        x_min, y_min, x_max, y_max = bbox_2d

        bboxes_2d.append([x_min, y_min, x_max, y_max])

    return np.array(bboxes_2d)