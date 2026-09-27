# DXA Quality Control AI

Локальный сервис для автоматической оценки качества DXA/DРА-исследований.

## Базовый контур
- поясничный отдел позвоночника (Spine);
- проксимальный отдел правого бедра (Right Hip);
- проксимальный отдел левого бедра (Left Hip).

## Текущая фаза проекта
Первый этап — аудит и нормализация обучающих данных. Исходные DICOM не изменяются.
XLSX содержит экспертные метки уровня исследования/папки. `dataset_manifest.csv` — это
технический индекс, связывающий конкретный DICOM с соответствующей строкой XLSX.

## Принцип архитектуры
Anatomy/Projection Classification -> Multi-task Quality Classification -> calibrated thresholds -> structured report.

Для визуального объяснения планируется Grad-CAM для каждого quality output.

## Структура данных для разработки
```text
project/
├── data/raw/                 # исходный датасет; не изменять
├── data/manifests/           # сгенерированные CSV/отчёты аудита
├── configs/labels.yaml       # правила комментариев/label engineering
├── app/data/build_manifest.py
├── app/data/comment_parser.py
├── app/dicom/loader.py
└── scripts/inspect_dataset.py
```

### Ожидаемый формат DICOM-папок
```text
001/
  spine.dcm
  left_hip.dcm
  right_hip.dcm
002/
  spine.dcm
  right_hip.dcm
...
```

Отсутствующий файл означает, что соответствующее исследование для этой папки отсутствует.

## Запуск подготовки данных
```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

python -m app.data.build_manifest \
  --dataset-root data/raw/dataset \
  --excel data/raw/labels.xlsx \
  --output data/manifests/dataset_manifest.csv \
  --warnings-output data/manifests/dataset_warnings.txt

python scripts/inspect_dataset.py data/manifests/dataset_manifest.csv
```

## Важно
- исходные DICOM и XLSX не редактируются программой;
- `manifest` можно пересоздавать детерминированно;
- персональные DICOM-поля не используются как ML-признаки;
- комментарии экспертов сначала превращаются в auxiliary tags; неоднозначные комментарии
  вроде `Требует внимание` не превращаются автоматически в конкретный quality target;
- split train/validation/test должен выполняться по `folder_id`, а не по отдельным изображениям.
