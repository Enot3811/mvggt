import cv2
import os
import argparse

def extract_frames(video_path, output_dir, step=1):
    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Создаем папку для сохранения кадров, если её не существует
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Видео: {video_path}")
    print(f"Всего кадров: {total_frames}")
    print(f"Шаг сохранения: {step}")
    print(f"Сохраняем картинки в папку: {output_dir}")

    current_frame = 0
    saved_count = 0

    while current_frame < total_frames:
        # Прыгаем на нужный кадр
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            print(f"Не удалось прочитать кадр {current_frame}, завершаем.")
            break

        # Генерируем имя файла с нулями в начале, чтобы они красиво сортировались
        # Например: frame_0000.jpg, frame_0005.jpg, frame_0120.jpg
        # Если кадров больше 10000, можно поменять 04d на 05d или 06d
        filename = f"frame_{current_frame:04d}.jpg"
        filepath = os.path.join(output_dir, filename)

        # Сохраняем кадр в файл
        cv2.imwrite(filepath, frame)
        saved_count += 1
        
        # Печатаем прогресс каждые 10 сохраненных кадров, чтобы было видно, что скрипт не завис
        if saved_count % 10 == 0:
            print(f"Сохранено {saved_count} кадров (обработан кадр {current_frame}/{total_frames-1})...")

        # Переходим к следующему нужному кадру
        current_frame += step

    cap.release()
    print(f"\nГотово! Успешно сохранено {saved_count} кадров в папку '{output_dir}'.")

def main():
    parser = argparse.ArgumentParser(description="Сохранение кадров из видео")
    parser.add_argument("video", help="Путь к видеофайлу")
    parser.add_argument("--output", "-o", default="extracted_frames", help="Папка для сохранения кадров (по умолчанию 'extracted_frames')")
    parser.add_argument("--step", "-s", type=int, default=1, help="Шаг извлечения кадров (по умолчанию 1)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.step)

if __name__ == "__main__":
    main()