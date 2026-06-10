import sys
import os
from pathlib import Path
import json
import csv
import numpy as np

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импортируем из analysis_core
from analysis_core import process_interview, load_audio, extract_paralinguistic


def calculate_cronbach_alpha(data_matrix):
    """
    Расчёт альфа Кронбаха вручную (без scipy).
    data_matrix: numpy array (n_samples, n_items)
    """
    n_samples, n_items = data_matrix.shape
    
    # Дисперсия каждого пункта
    item_variances = np.var(data_matrix, axis=0, ddof=1)
    
    # Общая дисперсия суммы
    total_variance = np.var(data_matrix.sum(axis=1), ddof=1)
    
    # Альфа Кронбаха
    alpha = (n_items / (n_items - 1)) * (1 - (item_variances.sum() / total_variance))
    return alpha


def run_experiments():
    """
    Основной скрипт экспериментов.
    Обрабатывает все аудиофайлы из папки data и сохраняет результаты.
    """
    data_dir = Path('data')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Создаём папку data, если её нет
    data_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ: Анализ аудиозаписей")
    print("="*60 + "\n")
    
    # Поиск всех аудиофайлов (mp3, m4a, aac, wav, ogg)
    audio_files = []
    for ext in ['*.mp3', '*.m4a', '*.aac', '*.wav', '*.ogg']:
        audio_files.extend(data_dir.glob(ext))
    audio_files = sorted(audio_files)
    
    if not audio_files:
        print(f"⚠ В папке {data_dir} не найдены аудиофайлы")
        print(f"💡 Поддерживаемые форматы: .mp3, .m4a, .aac, .wav, .ogg")
        print(f"💡 Поместите тестовые файлы в папку {data_dir}/")
        return
    
    print(f"Найдено {len(audio_files)} аудиофайлов. Начинаем обработку...\n")
    
    results = []
    
    for idx, audio_path in enumerate(audio_files, 1):
        print(f"[{idx}/{len(audio_files)}] Обработка {audio_path.name}...", end=' ')
        
        try:
            # Используем process_interview из analysis_core
            result = process_interview(str(audio_path))
            
            # Добавляем имя файла в результат
            result_row = {
                'file_name': audio_path.name,
                **result['domain_scores'],
                **{f'para_{k}': v for k, v in result['paralinguistic'].items()},
                'risk_level': result['risk_level'],
                'risk_score': result['risk_score'],
                'suicide_risk': result['suicide_risk_detected']
            }
            results.append(result_row)
            print("✓")
            
        except Exception as e:
            print(f"✗ ({e})")
            continue
    
    if not results:
        print("\n✗ Не удалось обработать ни один файл")
        return
    
    # ===== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ =====
    
    # CSV
    csv_path = results_dir / 'results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ CSV сохранён: {csv_path}")
    
    # JSON
    json_path = results_dir / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON сохранён: {json_path}")
    
    # ===== РАСЧЁТ АЛЬФА КРОНБАХА =====
    
    alpha_report = {}
    domain_names = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж']
    
    for domain in domain_names:
        domain_col = domain
        if domain_col in results[0]:
            values = np.array([r[domain_col] for r in results])
            alpha_report[domain] = {
                'mean': round(float(np.mean(values)), 3),
                'std': round(float(np.std(values)), 3),
                'min': round(float(np.min(values)), 3),
                'max': round(float(np.max(values)), 3),
                'n_samples': len(values)
            }
    
    # Расчёт альфа для всех доменов вместе
    all_domain_values = np.array([[r.get(d, 0) for d in domain_names] 
                                   for r in results])
    if all_domain_values.size > 0:
        overall_alpha = calculate_cronbach_alpha(all_domain_values)
        alpha_report['overall_alpha'] = round(overall_alpha, 3)
    
    alpha_path = results_dir / 'alpha_report.txt'
    with open(alpha_path, 'w', encoding='utf-8') as f:
        f.write("Коэффициент альфа Кронбаха и статистика доменов\n")
        f.write("="*60 + "\n\n")
        f.write(f"Всего обработано записей: {len(results)}\n\n")
        
        for domain, stats in alpha_report.items():
            if domain == 'overall_alpha':
                f.write(f"\nОбщая альфа Кронбаха (все домены): {stats:.3f}\n")
                f.write("  Интерпретация: ")
                if stats >= 0.9:
                    f.write("Отличная надёжность\n")
                elif stats >= 0.8:
                    f.write("Хорошая надёжность\n")
                elif stats >= 0.7:
                    f.write("Приемлемая надёжность\n")
                else:
                    f.write("Низкая надёжность\n")
            else:
                f.write(f"\nДомен {domain}:\n")
                f.write(f"  Среднее (Mean):    {stats['mean']:.3f}\n")
                f.write(f"  Ст. отклон (Std): {stats['std']:.3f}\n")
                f.write(f"  Минимум (Min):    {stats['min']:.3f}\n")
                f.write(f"  Максимум (Max):   {stats['max']:.3f}\n")
                f.write(f"  Кол-во образцов:  {stats['n_samples']}\n")
    
    print(f"✓ Отчёт альфа сохранён: {alpha_path}")
    
    # ===== ИТОГОВАЯ СТАТИСТИКА =====
    print("\n" + "="*60)
    print("ИТОГИ ЭКСПЕРИМЕНТОВ")
    print("="*60)
    print(f"Обработано записей: {len(results)}")
    print(f"Домены проанализированы: {', '.join(domain_names)}")
    if 'overall_alpha' in alpha_report:
        print(f"Альфа Кронбаха (все домены): {alpha_report['overall_alpha']:.3f}")
    print("\n✓ Все результаты сохранены в папку results/\n")


if __name__ == '__main__':
    run_experiments()