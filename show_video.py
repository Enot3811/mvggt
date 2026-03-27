import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description="Покадровый просмотр видео")
    parser.add_argument("video", help="Путь к видеофайлу (например, video.mp4)")
    parser.add_argument("--step", type=int, default=1, help="Шаг прокрутки кадров (по умолчанию 1)")
    args = parser.parse_args([
        "/home/enot/Загрузки/scenes/home.mp4",
        "--step",
        "60",
    ])

    # Открываем видеофайл
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видео {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    print("Управление:")
    print(f"  'd' - вперед на {args.step} кадр(ов)")
    print(f"  'a' - назад на {args.step} кадр(ов)")
    print("  'q' - выход")

    # Создаем окно, размер которого можно менять мышкой
    window_name = "Video Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        # Устанавливаем позицию на нужный кадр (OpenCV сделает это автоматически)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            print("\nДостигнут конец видео или ошибка чтения кадра.")
            # Если ушли за пределы, возвращаемся на последний кадр
            current_frame = max(0, total_frames - 1)
            continue

        # Добавляем текст поверх кадра для удобства
        info_text = f"Frame: {current_frame}/{total_frames-1} | Step: {args.step}"
        cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Показываем кадр
        cv2.imshow("Video Player", frame)

        # Ждем нажатия клавиши (0 означает бесконечное ожидание до первого нажатия)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'): # Выход
            break
        elif key == ord('d'): # Вперед
            current_frame = min(current_frame + args.step, total_frames - 1)
        elif key == ord('a'): # Назад
            current_frame = max(current_frame - args.step, 0)

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()