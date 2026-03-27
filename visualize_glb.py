import argparse
import trimesh
import sys

def visualize_glb(file_path):
    try:
        print(f"Загрузка файла: {file_path}")
        # Загружаем сцену/модель из файла
        scene = trimesh.load(file_path)
        
        print("Открытие интерактивного окна...")
        # Метод show() открывает окно визуализации на базе OpenGL
        # Вы можете крутить модель мышкой, приближать колесиком
        scene.show()
        
    except Exception as e:
        print(f"Ошибка при открытии файла: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Визуализатор 3D моделей в формате .glb")
    parser.add_argument(
        "file", 
        type=str, 
        help="Путь к файлу .glb (например: output/my_model.glb)"
    )
    
    args = parser.parse_args([
        "scene4.glb"
    ])
    visualize_glb(args.file)