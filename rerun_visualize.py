import numpy as np
import rerun as rr
import matplotlib.pyplot as plt
import argparse

def log_point_cloud(valid_points, valid_colors):
    """Логирует базовое 3D облако точек без выделения масок."""
    # Указываем, что базовая сцена относится к нулевому кадру маски, 
    # чтобы она попала в то же окно, что и камеры
    rr.set_time("mask_threshold", sequence=0)
    
    rr.log(
        "world/scene/point_cloud", 
        rr.Points3D(valid_points, colors=valid_colors, radii=0.01)
    )

def log_referring_mask(valid_points, valid_colors, mask, mask_logits, threshold=None):
    """
    Логирует 3D облако точек с наложенной маской (выделение красным).
    Если threshold указан - логирует одну статичную сцену.
    Если threshold=None - логирует 101 сцену в таймлайн для создания эффекта "ползунка".
    """
    # 1. Применяем Сигмоиду (переводим логиты в вероятности от 0.0 до 1.0)
    clipped_logits = np.clip(mask_logits, -100, 100)
    mask_probs = 1.0 / (1.0 + np.exp(-clipped_logits))
    
    if threshold is not None:
        # --- Статичный режим (Один фиксированный порог) ---
        rr.set_time("mask_threshold", sequence=0)
        
        current_colors = valid_colors.copy()
        highlight_indices = mask_probs[mask] > threshold
        current_colors[highlight_indices] = [255, 0, 0]
        
        rr.log(
            "world/scene/point_cloud", 
            rr.Points3D(valid_points, colors=current_colors, radii=0.01)
        )
    else:
        # --- Режим "Ползунка" (Таймлайн от 0.0 до 1.0) ---
        for i in range(0, 101, 5):
            t = i / 100.0
            
            # Сообщаем Rerun, что это "кадр", соответствующий порогу
            rr.set_time("mask_threshold", sequence=i)
            
            current_colors = valid_colors.copy()
            highlight_indices = mask_probs[mask] > t
            current_colors[highlight_indices] = [255, 0, 0]
            
            rr.log(
                "world/scene/point_cloud", 
                rr.Points3D(valid_points, colors=current_colors, radii=0.01)
            )

def log_cameras_and_heatmaps(num_frames, camera_poses, images, points, conf):
    """Логирует позиции камер (конусы), 2D-картинки и тепловые карты уверенности."""
    
    # Камеры и их картинки не должны зависеть от ползунка маски, 
    # поэтому мы жестко привязываем их к нулевому моменту на шкале mask_threshold
    rr.set_time("mask_threshold", sequence=0)
    
    cmap = plt.get_cmap("turbo") 
    
    for frame_idx in range(num_frames):
        cam_pose = camera_poses[frame_idx]
        translation = cam_pose[:3, 3]
        rotation_mat = cam_pose[:3, :3]
        
        cam_path = f"world/camera_{frame_idx}"
        
        # 1. Позиция камеры в 3D мире
        rr.log(cam_path, rr.Transform3D(translation=translation, mat3x3=rotation_mat))
        
        img_rgb = images[frame_idx]
        h, w = img_rgb.shape[:2]

        # 2. Логируем "Объектив" (Pinhole), чтобы связать 2D и 3D
        rr.log(cam_path, rr.Pinhole(resolution=[w, h], focal_length=w * 0.8))

        # 3. Логируем саму 2D картинку
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        rr.log(f"{cam_path}/image", rr.Image(img_rgb))

        # 4. Создаем и логируем тепловую карту уверенности (Confidence Heatmap)
        pts_cam = points[frame_idx].reshape(-1, 3)
        conf_cam = conf[frame_idx].reshape(-1)
        
        # Оставляем только точки с базовой уверенностью > 0.05
        valid_mask = conf_cam > 0.05
        pts_valid = pts_cam[valid_mask]
        conf_valid = conf_cam[valid_mask]
        
        # Переводим уверенность (от 0 до 1) в цвета (от синего к красному)
        colors_heatmap = (cmap(conf_valid)[:, :3] * 255).astype(np.uint8)
        
        rr.log(
            f"world/confidence_heatmap/camera_{frame_idx}",
            rr.Points3D(pts_valid, colors=colors_heatmap, radii=0.02)
        )

def visualize_with_rerun(npz_path, mask_threshold=None):
    """Основная функция для чтения данных и запуска Rerun."""
    print(f"Загрузка предсказаний из {npz_path}...")
    try:
        loaded = np.load(npz_path)
    except Exception as e:
        print(f"Ошибка загрузки файла: {e}")
        return

    # default_enabled=True сбрасывает старый интерфейс, чтобы Rerun собрал окна заново
    rr.init("MVGGT_Visualization", spawn=True, default_enabled=True)
    
    # 1. Достаем сырые данные
    points = loaded["points"]
    images = loaded["images"]
    conf = loaded["conf"]
    camera_poses = loaded["camera_poses"]
    
    # 2. Подготавливаем базовые точки (фильтрация по уверенности 0.1)
    # Это нужно, чтобы не рендерить миллионы "мусорных" точек из пустоты
    conf_threshold = 0.1 
    flat_points = points.reshape(-1, 3)
    flat_colors = images.copy().reshape(-1, 3)
    
    if flat_colors.max() <= 1.0:
        flat_colors = (flat_colors * 255).astype(np.uint8)
        
    flat_conf = conf.reshape(-1)
    mask = flat_conf > conf_threshold
    
    valid_points = flat_points[mask]
    valid_colors = flat_colors[mask]

    # 3. Логируем общую сцену (с маской объекта, если она есть)
    print("Логируем основное 3D облако...")
    if "referring_mask_pred" in loaded:
        flat_mask_logits = loaded["referring_mask_pred"].reshape(-1)
        log_referring_mask(
            valid_points, 
            valid_colors, 
            mask, 
            flat_mask_logits, 
            threshold=mask_threshold
        )
    else:
        log_point_cloud(valid_points, valid_colors)

    # 4. Логируем камеры и картинки
    print("Логируем камеры и тепловые карты...")
    log_cameras_and_heatmaps(points.shape[0], camera_poses, images.copy(), points, conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Визуализация предсказаний MVGGT в Rerun")
    
    # Путь до файла
    parser.add_argument(
        "npz_path", 
        type=str, 
        nargs='?', 
        default="/home/enot/Загрузки/scenes/scene1/predictions.npz", 
        help="Путь до файла predictions.npz"
    )
    
    # Опциональный фиксированный порог маски
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Фиксированный порог для маски (от 0.0 до 1.0). Если не указан, используется интерактивный ползунок-таймлайн."
    )
    
    args = parser.parse_args([
        "/home/enot/Загрузки/scenes/scene1/predictions.npz",
        "--threshold",
        "0.5",
    ])
    
    visualize_with_rerun(args.npz_path, mask_threshold=args.threshold)