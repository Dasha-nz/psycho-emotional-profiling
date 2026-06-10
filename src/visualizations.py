"""
Модуль визуализации результатов исследования
Генерирует графики для дипломной работы (без требований к статистической значимости)
"""

import sys
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
from matplotlib.collections import PatchCollection

# Настройка шрифтов для поддержки кириллицы
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Noto Sans']
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def load_results(csv_path):
    """Загрузка результатов из CSV"""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({k: float(v) if _is_float(v) else v
                        for k, v in row.items()})
    return data


def _is_float(val):
    """Проверка, является ли строка числом"""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


# ============================================================
# АРХИТЕКТУРНЫЕ СХЕМЫ (не требуют данных)
# ============================================================

def plot_architecture():
    """График 1: Архитектура системы (воронка)"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Заголовок
    ax.text(7, 9.5, 'Архитектура системы психоэмоционального профилирования', 
            fontsize=14, fontweight='bold', ha='center', va='top')

    # === ВХОД (верх) ===
    ax.text(7, 8.7, 'Входные данные', fontsize=11, fontweight='bold', ha='center')
    input_box = FancyBboxPatch((5.5, 8.2), 3, 0.6, boxstyle='round,pad=0.1', 
                               linewidth=2, edgecolor='#1565C0', facecolor='#E3F2FD')
    ax.add_patch(input_box)
    ax.text(7, 8.5, 'Аудиофайл\n(mp3, m4a, aac, wav, ogg)', 
            fontsize=10, ha='center', va='center', fontweight='bold')

    # === РАЗДЕЛЕНИЕ НА ДВА ПОТОКА ===

    # Левая ветка — Лексический анализ
    ax.text(3.5, 7.3, 'Лингвистический анализ', fontsize=11, fontweight='bold', 
            ha='center', color='#0D47A1')

    # Whisper
    whisper_box = FancyBboxPatch((1.5, 6.3), 4, 0.7, boxstyle='round,pad=0.1', 
                                  linewidth=1.5, edgecolor='#1976D2', facecolor='#BBDEFB')
    ax.add_patch(whisper_box)
    ax.text(3.5, 6.65, 'Whisper\n(распознавание речи → текст)', 
            fontsize=9, ha='center', va='center')

    # Лексический анализ
    lexical_box = FancyBboxPatch((1.5, 5.1), 4, 0.7, boxstyle='round,pad=0.1', 
                                  linewidth=1.5, edgecolor='#1976D2', facecolor='#90CAF9')
    ax.add_patch(lexical_box)
    ax.text(3.5, 5.45, 'Лексический анализ\n(7 доменов: А-Ж)', 
            fontsize=9, ha='center', va='center')

    # Домены (список)
    domains_text = 'Домены:\n• А — Факторологический\n• Б — Аффективный\n• В — Соматический\n• Г — Когнитивный\n• Д — Тревожный\n• Е — Самооценка\n• Ж — Завершающий'
    ax.text(3.5, 3.5, domains_text, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1))

    # Правая ветка — Паралингвистический анализ
    ax.text(10.5, 7.3, 'Паралингвистический анализ', fontsize=11, fontweight='bold', 
            ha='center', color='#1B5E20')

    # Извлечение признаков
    paraling_box = FancyBboxPatch((8.5, 5.1), 4, 1.9, boxstyle='round,pad=0.1', 
                                   linewidth=1.5, edgecolor='#388E3C', facecolor='#C8E6C9')
    ax.add_patch(paraling_box)
    ax.text(10.5, 6.4, 'Извлечение признаков\nиз аудиосигнала', 
            fontsize=9, ha='center', va='center', fontweight='bold')

    paraling_features = '• Темп речи (слов/мин)\n• Доля пауз (%)\n• Вариативность тона (полутоны)\n• Jitter (%)\n• Shimmer (%)\n• Интенсивность (dB)'
    ax.text(10.5, 5.5, paraling_features, fontsize=8, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1))

    # === СХОДИМОСТЬ ВНИЗ (воронка) ===
    ax.arrow(3.5, 2.8, 0, -0.8, head_width=0.4, head_length=0.5, fc='#1565C0', ec='#1565C0', linewidth=2)
    ax.arrow(10.5, 2.8, 0, -0.8, head_width=0.4, head_length=0.5, fc='#2E7D32', ec='#2E7D32', linewidth=2)

    # === ОБЪЕДИНЕНИЕ ===
    merge_box = FancyBboxPatch((5.5, 1.3), 3, 0.7, boxstyle='round,pad=0.1', 
                                linewidth=2, edgecolor='#6A1B9A', facecolor='#E1BEE7')
    ax.add_patch(merge_box)
    ax.text(7, 1.65, 'Объединение результатов\n+ расчёт индикаторов', 
            fontsize=10, ha='center', va='center', fontweight='bold')

    # Стрелка вниз
    ax.arrow(7, 1.3, 0, -0.5, head_width=0.4, head_length=0.5, fc='#6A1B9A', ec='#6A1B9A', linewidth=2)

    # === ВЫХОД ===
    output_box = FancyBboxPatch((5.5, 0.3), 3, 0.7, boxstyle='round,pad=0.1', 
                                 linewidth=2, edgecolor='#4A148C', facecolor='#CE93D8')
    ax.add_patch(output_box)
    ax.text(7, 0.65, 'Итоговый отчёт\n(оценки, риски, рекомендации)', 
            fontsize=10, ha='center', va='center', fontweight='bold')

    # === СТРЕЛКИ ОТ ВХОДА К ВЕТКАМ ===
    ax.arrow(6.2, 8.5, 1.5, -0.8, head_width=0.3, head_length=0.4, fc='#757575', ec='#757575', linewidth=1.5, linestyle='--')
    ax.arrow(7.8, 8.5, -1.5, -0.8, head_width=0.3, head_length=0.4, fc='#757575', ec='#757575', linewidth=1.5, linestyle='--')

    plt.tight_layout()
    output = Path('results') / 'plot_architecture_v2.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ График 1 (Архитектура): {output}")
    plt.close()


def plot_processing_time():
    """График 2: Производительность системы"""
    audio_duration = [45, 78]
    processing_time = [62, 105]

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(audio_duration, processing_time, s=100, color='steelblue', alpha=0.7)
    plt.plot(audio_duration, processing_time, 'b-', linewidth=2, alpha=0.5)

    max_time = max(max(audio_duration), max(processing_time))
    plt.plot([0, max_time], [0, max_time], 'r--', linewidth=1.5, alpha=0.5, label='Real-time (1:1)')

    plt.xlabel('Длительность аудио (сек)', fontsize=11, fontweight='bold')
    plt.ylabel('Время обработки (сек)', fontsize=11, fontweight='bold')
    plt.title('Производительность системы обработки аудио', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    for i, (dur, proc) in enumerate(zip(audio_duration, processing_time)):
        plt.annotate(f'Запись {i+1}\n({dur} сек → {proc} сек)', 
                    (dur, proc), xytext=(10, 10), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output = Path('results') / 'plot_processing_time.png'
    plt.savefig(output, dpi=300)
    print(f"✓ График 2 (Производительность): {output}")
    plt.close()


# ============================================================
# ИНДИВИДУАЛЬНЫЙ АНАЛИЗ (case studies — не статистика)
# ============================================================

def plot_individual_case_study(results):
    """График 3: Детальный разбор индивидуальных случаев (case study)"""
    if not results:
        print("⚠ Нет данных для индивидуального анализа")
        return
    
    domains = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж']
    domain_names = {
        'А': 'Фактороло-\nгический',
        'Б': 'Аффективно-\nэмоциональный',
        'В': 'Сомато-\nвегетативный',
        'Г': 'Когни-\nтивный',
        'Д': 'Тревожный',
        'Е': 'Само-\nоценка',
        'Ж': 'Завер-\nшающий'
    }
    
    # Создаём подграфики для каждой записи
    n_records = min(len(results), 3)  # Показываем максимум 3 записи
    fig, axes = plt.subplots(1, n_records, figsize=(6*n_records, 6))
    if n_records == 1:
        axes = [axes]
    
    for idx in range(n_records):
        ax = axes[idx]
        result = results[idx]
        
        values = [result.get(d, 0) for d in domains]
        colors = ['#2E7D32' if v >= 0 else '#C62828' for v in values]  # Зелёный/Красный
        
        bars = ax.bar(range(len(domains)), values, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
        
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels([domain_names[d] for d in domains], fontsize=9, rotation=45, ha='right')
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Оценка (-1.0 до +1.0)', fontsize=10, fontweight='bold')
        ax.set_title(f'Кейс {idx+1}: {result.get("file_name", "Запись")}', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Добавляем значения на столбцы
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.05 if val >= 0 else -0.15),
                   f'{val:+.2f}', ha='center', va='bottom' if val >= 0 else 'top', 
                   fontsize=9, fontweight='bold')
        
        # Добавляем информацию о риске
        risk_level = result.get('risk_level', 'N/A')
        risk_score = result.get('risk_score', 0)
        ax.text(0.95, 0.95, f'Риск: {risk_level}\n(балл: {risk_score})',
               transform=ax.transAxes, fontsize=9, fontweight='bold',
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Индивидуальный психоэмоциональный профиль (case study)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output = Path('results') / 'plot_individual_case_study.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ График 3 (Case study): {output}")
    plt.close()


def plot_paralinguistic_radar(results):
    """График 4: Radar chart паралингвистических параметров для каждого кейса"""
    if not results:
        print("⚠ Нет данных для radar chart")
        return
    
    # Параметры для radar chart
    params = ['speech_rate_wpm', 'pause_ratio_pct', 'f0_variability_semitones', 
              'jitter_pct', 'shimmer_pct', 'intensity_db']
    param_names = ['Темп речи\n(слов/мин)', 'Доля пауз\n(%)', 'Вариативность тона\n(полутоны)',
                   'Jitter\n(%)', 'Shimmer\n(%)', 'Интенсивность\n(dB)']
    
    # Нормализация для radar chart (приводим к шкале 0-1)
    norms = {
        'speech_rate_wpm': (80, 200),
        'pause_ratio_pct': (0, 50),
        'f0_variability_semitones': (0, 10),
        'jitter_pct': (0, 5),
        'shimmer_pct': (0, 10),
        'intensity_db': (30, 90)
    }
    
    n_records = min(len(results), 3)
    fig, axes = plt.subplots(1, n_records, figsize=(6*n_records, 6), subplot_kw=dict(polar=True))
    if n_records == 1:
        axes = [axes]
    
    angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
    angles += angles[:1]  # Замыкаем круг
    
    for idx in range(n_records):
        ax = axes[idx]
        result = results[idx]
        
        # Нормализуем значения
        values = []
        for param in params:
            val = result.get(f'para_{param}', 0)
            min_val, max_val = norms[param]
            normalized = (val - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0
            normalized = max(0, min(1, normalized))  # Ограничиваем 0-1
            values.append(normalized)
        
        values += values[:1]  # Замыкаем круг
        
        # Рисуем radar chart
        ax.plot(angles, values, 'o-', linewidth=2, markersize=6, color='#1976D2')
        ax.fill(angles, values, alpha=0.25, color='#1976D2')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(param_names, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f'Кейс {idx+1}: {result.get("file_name", "Запись")}', 
                    fontsize=11, fontweight='bold', pad=20)
        ax.grid(True)
    
    plt.suptitle('Паралингвистические параметры (нормализованная шкала 0-1)', 
                fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    output = Path('results') / 'plot_paralinguistic_radar.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ График 4 (Radar chart): {output}")
    plt.close()


# ============================================================
# МЕТОДОЛОГИЯ И СТРУКТУРА
# ============================================================

def plot_interview_structure():
    """График 5: Структура интервью по этапам"""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Заголовок
    ax.text(5, 6.5, 'Структура структурированного интервью', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Этап 1
    ax.add_patch(FancyBboxPatch((1, 5.2), 3, 0.8, boxstyle='round,pad=0.1', 
                                 linewidth=2, edgecolor='#1565C0', facecolor='#E3F2FD'))
    ax.text(2.5, 5.6, 'Этап 1: Калибровка', fontsize=11, fontweight='bold', ha='center')
    ax.text(2.5, 5.3, '3-4 минуты • 7 вопросов (К1-К7)', fontsize=9, ha='center')
    
    # Этап 2
    ax.add_patch(FancyBboxPatch((1, 3.8), 3, 0.8, boxstyle='round,pad=0.1', 
                                 linewidth=2, edgecolor='#388E3C', facecolor='#E8F5E9'))
    ax.text(2.5, 4.2, 'Этап 2: Основное интервью', fontsize=11, fontweight='bold', ha='center')
    ax.text(2.5, 3.9, '10-15 минут • 7 доменов (А-Ж)', fontsize=9, ha='center')
    
    # Этап 3
    ax.add_patch(FancyBboxPatch((1, 2.4), 3, 0.8, boxstyle='round,pad=0.1', 
                                 linewidth=2, edgecolor='#F57C00', facecolor='#FFF3E0'))
    ax.text(2.5, 2.8, 'Этап 3: Валидация', fontsize=11, fontweight='bold', ha='center')
    ax.text(2.5, 2.5, '1-2 минуты • Повтор К1, К4, К5', fontsize=9, ha='center')
    
    # Домены (справа)
    domains_text = 'Основные домены:\n• А — Факторологический\n• Б — Аффективно-эмоциональный\n• В — Сомато-вегетативный\n• Г — Когнитивный\n• Д — Тревожный\n• Е — Самооценка и риск\n• Ж — Завершающий'
    ax.add_patch(FancyBboxPatch((6, 2.4), 3.5, 3.6, boxstyle='round,pad=0.1', 
                                 linewidth=1.5, edgecolor='#7B1FA2', facecolor='#F3E5F5'))
    ax.text(7.75, 4.4, domains_text, fontsize=9, ha='center', va='center')
    
    # Стрелки
    ax.arrow(4, 5.6, 1.5, 0, head_width=0.3, head_length=0.4, fc='#757575', ec='#757575', linewidth=1.5)
    ax.arrow(4, 4.2, 1.5, 0, head_width=0.3, head_length=0.4, fc='#757575', ec='#757575', linewidth=1.5)
    
    plt.tight_layout()
    output = Path('results') / 'plot_interview_structure.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ График 5 (Структура интервью): {output}")
    plt.close()


def plot_methodology_flowchart():
    """График 6: Блок-схема методологии анализа"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Заголовок
    ax.text(6, 9.5, 'Методология анализа психоэмоционального состояния', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Вход
    ax.add_patch(FancyBboxPatch((4.5, 8.3), 3, 0.6, boxstyle='round,pad=0.1', 
                                 linewidth=2, edgecolor='#1565C0', facecolor='#E3F2FD'))
    ax.text(6, 8.6, 'Аудиозапись интервью', fontsize=10, fontweight='bold', ha='center')
    
    # Распознавание
    ax.add_patch(FancyBboxPatch((4.5, 7.2), 3, 0.6, boxstyle='round,pad=0.1', 
                                 linewidth=1.5, edgecolor='#1976D2', facecolor='#BBDEFB'))
    ax.text(6, 7.5, 'Whisper: речь → текст', fontsize=9, ha='center')
    
    # Два потока
    ax.text(3, 6.3, 'Вербальный анализ', fontsize=10, fontweight='bold', ha='center', color='#0D47A1')
    ax.text(9, 6.3, 'Паралингвистика', fontsize=10, fontweight='bold', ha='center', color='#1B5E20')
    
    # Лексический анализ
    ax.add_patch(FancyBboxPatch((1, 5), 3, 0.8, boxstyle='round,pad=0.1', 
                                 linewidth=1.5, edgecolor='#1976D2', facecolor='#90CAF9'))
    ax.text(2.5, 5.6, 'Лексический анализ', fontsize=9, fontweight='bold', ha='center')
    ax.text(2.5, 5.3, 'NEGATIVE/POSITIVE\nлексиконы', fontsize=8, ha='center')
    
    # Паралингвистика
    ax.add_patch(FancyBboxPatch((8, 5), 3, 0.8, boxstyle='round,pad=0.1', 
                                 linewidth=1.5, edgecolor='#388E3C', facecolor='#C8E6C9'))
    ax.text(9.5, 5.6, 'Акустические признаки', fontsize=9, fontweight='bold', ha='center')
    ax.text(9.5, 5.3, 'Темп, паузы, тон,\njitter, shimmer', fontsize=8, ha='center')
    
    # Домены
    ax.add_patch(FancyBboxPatch((1, 3.3), 3, 0.8, boxstyle='round,pad=0.1', 
                                 linewidth=1.5, edgecolor='#1976D2', facecolor='#64B5F6'))
    ax.text(2.5, 3.9, '7 доменов (А-Ж)', fontsize=9, fontweight='bold', ha='center')
    ax.text(2.5, 3.6, 'Шкала: -1.0 до +1.0', fontsize=8, ha='center')
    
    # Интеграция
    ax.add_patch(FancyBboxPatch((8, 3.3), 3, 0.8, boxstyle='round,pad=0.1', 
                                 linewidth=1.5, edgecolor='#388E3C', facecolor='#81C784'))
    ax.text(9.5, 3.9, 'Интеграция признаков', fontsize=9, fontweight='bold', ha='center')
    ax.text(9.5, 3.6, 'Валидация пространства', fontsize=8, ha='center')
    
    # Результат
    ax.add_patch(FancyBboxPatch((4.5, 1.8), 3, 0.8, boxstyle='round,pad=0.1', 
                                 linewidth=2, edgecolor='#6A1B9A', facecolor='#E1BEE7'))
    ax.text(6, 2.4, 'Итоговый отчёт', fontsize=10, fontweight='bold', ha='center')
    ax.text(6, 2.1, 'Оценки, риск, рекомендации', fontsize=8, ha='center')
    
    # Стрелки
    ax.arrow(6, 8.3, 0, -0.5, head_width=0.3, head_length=0.4, fc='#757575', ec='#757575', linewidth=1.5)
    ax.arrow(6, 7.2, 0, -0.4, head_width=0.3, head_length=0.4, fc='#757575', ec='#757575', linewidth=1.5)
    ax.arrow(4.5, 7, -1.2, -0.5, head_width=0.25, head_length=0.3, fc='#757575', ec='#757575', linewidth=1.2, linestyle='--')
    ax.arrow(7.5, 7, 1.2, -0.5, head_width=0.25, head_length=0.3, fc='#757575', ec='#757575', linewidth=1.2, linestyle='--')
    ax.arrow(2.5, 5, 0, -0.9, head_width=0.3, head_length=0.4, fc='#1565C0', ec='#1565C0', linewidth=1.5)
    ax.arrow(9.5, 5, 0, -0.9, head_width=0.3, head_length=0.4, fc='#2E7D32', ec='#2E7D32', linewidth=1.5)
    ax.arrow(4, 3.7, 2, -0.5, head_width=0.25, head_length=0.3, fc='#757575', ec='#757575', linewidth=1.2, linestyle='--')
    ax.arrow(8, 3.7, -2, -0.5, head_width=0.25, head_length=0.3, fc='#757575', ec='#757575', linewidth=1.2, linestyle='--')
    ax.arrow(6, 3.3, 0, -0.7, head_width=0.3, head_length=0.4, fc='#6A1B9A', ec='#6A1B9A', linewidth=1.5)
    
    plt.tight_layout()
    output = Path('results') / 'plot_methodology_flowchart.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✓ График 6 (Методология): {output}")
    plt.close()


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def generate_all_visualizations():
    """Главная функция — генерирует все графики"""
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ ВИЗУАЛИЗАЦИЙ (без статистических требований)")
    print("="*60 + "\n")
    
    Path('results').mkdir(exist_ok=True)
    
    # Загружаем данные если есть
    csv_path = Path('results') / 'results.csv'
    results = []
    if csv_path.exists():
        print("Загрузка данных из results.csv...")
        results = load_results(str(csv_path))
        print(f"Загружено {len(results)} записей\n")
    
    # Генерируем все графики
    print("Генерация визуализаций...")
    plot_architecture()
    plot_processing_time()
    
    if results:
        plot_individual_case_study(results)
        plot_paralinguistic_radar(results)
    else:
        print("⚠ Пропускаем графики с данными (нет results.csv)")
    
    plot_interview_structure()
    plot_methodology_flowchart()
    
    print("\n" + "="*60)
    print("✓ Все визуализации сгенерированы!")
    print("="*60)
    print("\nРасположение файлов:")
    print("  results/plot_architecture_v2.png          (архитектура)")
    print("  results/plot_processing_time.png          (производительность)")
    if results:
        print("  results/plot_individual_case_study.png    (case study)")
        print("  results/plot_paralinguistic_radar.png     (radar chart)")
    print("  results/plot_interview_structure.png      (структура интервью)")
    print("  results/plot_methodology_flowchart.png    (методология)\n")


if __name__ == '__main__':
    generate_all_visualizations()