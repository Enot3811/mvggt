**User**

Можешь выполнить ещё одно задание? Посмотри, что можно залогировать в ходе обучения с помощью wandb. Начни отсюда @trainers/base_trainer_accelerate.py 

---

**Cursor**

Основываясь на коде `trainers/base_trainer_accelerate.py`, `trainers/mvggt_trainer.py` и `mvggt/models/loss.py`, система логирования через **wandb** (а также TensorBoard, если он выбран) настроена очень подробно.

Вот полный список того, что логируется в процессе обучения:

### 1. Метрики шагов обучения (логируются каждые 10 шагов)
Во время `train_one_epoch` модель собирает все компоненты лосса и различные системные метрики:
* **Лоссы геометрии и камеры** (из `MVGGTLoss` и `PointLoss`):
  * `train/loss` — общий итоговый лосс.
  * `train/local_pts_loss` и `train/global_pts_loss` — L1 потери для предсказания точек.
  * `train/normal_loss` — потери по нормалям.
  * `train/trans_loss` и `train/rot_loss` — потери предсказания позы камеры (смещение и поворот).
* **Лоссы сегментации 2D масок** (из `ReferringMaskLoss`):
  * `train/refer_loss_mask` — тот самый 2D BCE Loss.
  * `train/refer_loss_dice` — PVSO Dice Loss.
  * `train/refer_iou_score` — тренировочный IoU.
  * Если используются промежуточные слои (layer-wise), то логируются `train/refer_loss_mask_layer_X`, `train/refer_loss_dice_layer_X` и т.д.
* **Статистика по батчу**:
  * `train/refer_rate_no_target` — доля батча, в которой объект отсутствует во всех кадрах.
  * `train/refer_rate_frame_with_target` — доля кадров, на которых объект виден.
  * `train/refer_rate_pixel_with_target` — средний размер объекта (в долях пикселей).
* **Системные метрики градиентов и оптимизатора**:
  * `lr` и `min_lr` — текущий Learning Rate.
  * `grad_norm` — норма градиентов (помогает отследить взрыв градиентов).
  * `weight_decay`.

### 2. Метрики валидации (в конце каждой эпохи)
По завершении каждой эпохи запускается метод `validate()`, который прогоняет тестовый датасет и вычисляет огромное количество метрик, добавляя к ним префикс `val_`.

В первую очередь логируются **базовые IoU метрики**:
* **Per-view метрики** (усредненные по отдельным 2D кадрам):
  * `val_Acc@0.25_per_view_overall`, `val_Acc@0.50_per_view_overall` — Accuracy при пороге IoU 0.25 и 0.5.
  * `val_iou_per_view_overall` — среднее значение IoU по 2D кадрам.
* **Global метрики** (оценка маски в целом для всего сэмпла):
  * `val_Acc@0.25_global_overall`, `val_Acc@0.50_global_overall`
  * `val_iou_global_overall`

Далее эти же метрики дробятся на **подкатегории**:
1. **По сложности (Hard / Easy)**. Вычисляется на основе пикселей объекта. Если на кадрах объект занимает очень малую площадь (< 5%), сэмпл считается Hard:
   * `val_Acc@0.25_global_hard` / `_easy`
   * `val_iou_global_hard` / `_easy`
   * (и аналогично для per-view).
2. **ScanRefer-специфичные метрики (Unique / Multiple)**. Логируются, если загружен файл `data/lookup.json` (когда в сцене только один объект такого класса — Unique, или их несколько — Multiple):
   * `val_iou_global_unique` / `val_iou_global_multiple` и их Accuracy.
3. **Nr3D / Sr3D-специфичные метрики** (если используется датасет ReferIt3D):
   * `val_ReferIt3D_mIoU_Hard` / `_Easy` (основано на количестве объектов-дистракторов в метаданных).
   * `val_ReferIt3D_mIoU_ViewDep` / `_ViewIndep` (зависит ли текстовый запрос от положения камеры).

### 3. Логирование изображений (Images)
В методе `log_all` (строки 1170-1172 в `base_trainer_accelerate.py`) есть проверка:
```python
if Image.isImageType(v):
    log_img[prefix+'/'+k] = v
# ...
for tracker in self.accelerator.trackers:
    tracker.log_images(log_img, step)
```
Это означает, что wandb полностью готов к логированию картинок. Если в методе подсчёта лосса (или метрик) добавить в возвращаемый словарь `details` объекты типа `PIL.Image`, то wandb автоматически подхватит их и отобразит в панели (например, предсказанные маски vs GT маски для визуальной оценки в ходе обучения). На данный момент в самом лоссе этого нет, но архитектура трей трейнера это заложена позволяет добавить это парой строк кода.

