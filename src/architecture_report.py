import whisper
import torch
from torchinfo import summary
import os

# Создаём папку docs, если её нет
os.makedirs('docs', exist_ok=True)

# Загружаем модель
model = whisper.load_model('small')
encoder = model.encoder  # часть модели, обрабатывающая звук

# Whisper принимает мел-спектрограмму 80 x 3000
dummy_input = torch.zeros(1, 80, 3000)

# Генерируем отчёт
report = summary(encoder, input_data=dummy_input, depth=4, verbose=0)

# Сохраняем в файл
with open('docs/architecture_report.txt', 'w', encoding='utf-8') as f:
    f.write(str(report))

print('Отчёт сохранён в docs/architecture_report.txt')