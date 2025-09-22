import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import json

# Предварительно загружаем и создаем mapping
with open('.\\camera_lidar_semantic_bboxes\\class_list.json', 'r') as f:
    color_to_class_name = json.load(f)

# Группируем классы по категориям (твое первое сокращение)
class_grouping = {
    # Основные классы дорожной сцены
    'road': ['RD normal street', 'Drivable cobblestone', 'Non-drivable street'],
    'sidewalk': ['Sidewalk', 'Curbstone'],
    'building': ['Buildings'],
    'vegetation': ['Nature object'],
    'sky': ['Sky'],
    
    # Транспорт
    'car': ['Car 1', 'Car 2', 'Car 3', 'Car 4', 'Ego car'],
    'truck': ['Truck 1', 'Truck 2', 'Truck 3'],
    'bicycle': ['Bicycle 1', 'Bicycle 2', 'Bicycle 3', 'Bicycle 4'],
    'motorcycle': ['Small vehicles 1', 'Small vehicles 2', 'Small vehicles 3'],
    'bus': ['Utility vehicle 1', 'Utility vehicle 2'],
    
    # Люди
    'person': ['Pedestrian 1', 'Pedestrian 2', 'Pedestrian 3'],
    
    # Дорожная инфраструктура
    'traffic_light': ['Traffic signal 1', 'Traffic signal 2', 'Traffic signal 3', 'Electronic traffic'],
    'traffic_sign': ['Traffic sign 1', 'Traffic sign 2', 'Traffic sign 3', 'Irrelevant signs'],
    'pole': ['Poles', 'Signal corpus'],
    'fence': ['Sidebars', 'Road blocks'],
    
    # Разметка
    'line': ['Solid line', 'Dashed line', 'Painted driv. instr.'],
    'crosswalk': ['Zebra crossing'],
    
    # Прочее
    'terrain': ['Parking area', 'Slow drive area', 'Grid structure'],
    'unknown': ['Blurred area', 'Rain dirt', 'Obstacles / trash', 'Animals', 'Tractor']
}

# Создаем mapping из цветов в class_id
color_to_class_id = {}
class_id_to_name = {}

# Сначала создаем mapping для всех оригинальных цветов
for hex_code, class_name in color_to_class_name.items():
    hex_color = hex_code.lstrip('#')
    rgb = tuple(int(hex_color[j:j+2], 16) for j in (0, 2, 4))
    bgr = (rgb[2], rgb[1], rgb[0])  # OpenCV uses BGR
    color_to_class_id[bgr] = class_name

# Теперь создаем mapping для сокращенных классов
simplified_color_to_class_id = {}
simplified_class_mapping = {}

# Создаем mapping из старых имен в новые
for new_class, old_classes in class_grouping.items():
    for old_class in old_classes:
        simplified_class_mapping[old_class] = new_class

# Создаем числовые ID для новых классов
unique_new_classes = sorted(set(simplified_class_mapping.values()))
new_class_to_id = {cls: i for i, cls in enumerate(unique_new_classes)}
id_to_new_class = {i: cls for i, cls in enumerate(unique_new_classes)}

print(f"Сокращено до {len(unique_new_classes)} классов:")
for class_id, class_name in id_to_new_class.items():
    print(f"  {class_id}: {class_name}")

# Функция для преобразования цветной маски в классы
def convert_color_mask_to_classes(color_mask, color_to_class_id, simplified_class_mapping, new_class_to_id):
    height, width, _ = color_mask.shape
    class_mask = np.zeros((height, width), dtype=np.int64)
    
    for i in range(height):
        for j in range(width):
            pixel_color = tuple(color_mask[i, j])
            if pixel_color in color_to_class_id:
                original_class = color_to_class_id[pixel_color]
                if original_class in simplified_class_mapping:
                    new_class = simplified_class_mapping[original_class]
                    class_mask[i, j] = new_class_to_id[new_class]
                else:
                    class_mask[i, j] = new_class_to_id['unknown']  # fallback
            else:
                class_mask[i, j] = new_class_to_id['unknown']  # unknown color
    
    return class_mask

class SegmentationDataset(Dataset):
    def __init__(self, image_files, mask_files, image_size=(300, 300)):
        self.image_files = image_files
        self.mask_files = mask_files
        self.image_size = image_size
        self.num_classes = len(unique_new_classes)

        # Transform для изображений
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        # Transform для масок (только resize)
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.NEAREST),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Загружаем изображение
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Загружаем маску
        mask_path = self.mask_files[idx]
        color_mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        
        # Преобразуем цветную маску в классы
        class_mask = convert_color_mask_to_classes(
            color_mask, color_to_class_id, simplified_class_mapping, new_class_to_id
        )
        
        # Конвертируем в PIL для трансформаций
        class_mask_pil = Image.fromarray(class_mask.astype(np.uint8))
        
        # Применяем трансформации
        img_transformed = self.image_transform(img)
        mask_transformed = self.mask_transform(class_mask_pil)
        
        # Конвертируем маску обратно в тензор
        mask_transformed = torch.from_numpy(np.array(mask_transformed)).long()
        
        return img_transformed, mask_transformed

    def get_num_classes(self):
        return self.num_classes

# Collate function для DataLoader
def collate_fn(batch):
    images = []
    masks = []
    
    for img, mask in batch:
        images.append(img)
        masks.append(mask)
    
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    
    return images, masks

class UltraFastSegmentationDataset(Dataset):
    def __init__(self, image_files, preprocessed_mask_files, image_size=(300, 300)):
        self.image_files = image_files
        self.preprocessed_mask_files = preprocessed_mask_files
        self.image_size = image_size
        self.num_classes = len(unique_new_classes)

        # Transform для изображений (маски уже с resize)
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Загружаем изображение
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Загружаем ПРЕДОБРАБОТАННУЮ маску (уже с resize)
        mask_path = self.preprocessed_mask_files[idx]
        mask_transformed = np.load(mask_path)
        
        # Преобразуем изображение
        img_transformed = self.image_transform(img)
        
        # Конвертируем маску в тензор
        mask_transformed = torch.from_numpy(mask_transformed).long()
        
        return img_transformed, mask_transformed