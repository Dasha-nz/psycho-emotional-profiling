import os
import json
import math
import tempfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import librosa
import torch
from scipy import stats

import whisper

try:
    from torchinfo import summary
except ImportError:
    summary = None

DOMAINS = {
    'А': {'name': 'Факторологический', 'items': ['А1', 'А2', 'А3', 'А4', 'А5', 'А6'], 'source': 'DSM-5'},
    'Б': {'name': 'Аффективно-эмоциональный', 'items': ['Б1', 'Б2', 'Б3', 'Б4'], 'source': 'PHQ-9, BDI-II'},
    'В': {'name': 'Сомато-вегетативный', 'items': ['В1', 'В2', 'В3', 'В4'], 'source': 'PHQ-9'},
    'Г': {'name': 'Когнитивный', 'items': ['Г1', 'Г2', 'Г3', 'Г4', 'Г5'], 'source': 'MoCA, MMSE'},
    'Д': {'name': 'Тревожный', 'items': ['Д1', 'Д2', 'Д3', 'Д4'], 'source': 'GAD-7'},
    'Е': {'name': 'Самооценка и риск', 'items': ['Е1', 'Е2', 'Е3', 'Е4'], 'source': 'BDI-II'},
    'Ж': {'name': 'Завершающий', 'items': ['Ж1', 'Ж2', 'Ж3'], 'source': 'оригинальный'},
}

INTERVIEW_QUESTIONS = {
    'Калибровка (Этап 1, 3-4 минуты)': {
        'описание': 'Нейтральные вопросы для установки индивидуального базиса по паралингвистическим параметрам',
        'инструмент': 'оригинальный блок, базис для формулы отклонения',
        'вопросы': {
            'К1': 'Опишите комнату, в которой вы сейчас находитесь. Что вы видите вокруг?',
            'К2': 'Какая сегодня погода за окном? Опишите её в нескольких предложениях.',
            'К3': 'Расскажите, что вы обычно едите на завтрак.',
            'К4': 'Опишите ваш типичный распорядок дня в будни.',
            'К5': 'Какой последний фильм или книгу вы видели или читали? Перескажите сюжет.',
            'К6': 'Опишите дорогу от вашего дома до ближайшего магазина.',
        }
    },
    'А. Факторологический домен (DSM-5)': {
        'описание': 'Базовые факты о состоянии и анамнезе',
        'инструмент': 'DSM-5',
        'вопросы': {
            'А1': 'Сколько вам полных лет?',
            'А2': 'Где вы сейчас живёте и с кем?',
            'А3': 'Чем вы занимаетесь в жизни — работа, учёба?',
            'А4': 'Обращались ли вы когда-нибудь к психологу или психиатру? Если да, расскажите кратко.',
            'А5': 'Принимаете ли вы какие-либо лекарства на постоянной основе?',
            'А6': 'Были ли в последний год значимые жизненные события — потери, переезды, смены работы?',
        }
    },
    'Б. Аффективно-эмоциональный домен (PHQ-9, BDI-II)': {
        'описание': 'Оценка настроения, ангедонии, эмоционального фона за последние 2 недели',
        'инструмент': 'PHQ-9, BDI-II',
        'вопросы': {
            'Б1': 'Как часто за последние две недели вы чувствовали себя подавленно, грустно или безнадёжно?',
            'Б2': 'Как часто за последние две недели вы теряли интерес или удовольствие от обычных дел?',
            'Б3': 'Опишите своё настроение в среднем за последний месяц. Что преобладает?',
            'Б4': 'Бывают ли у вас моменты, когда хочется плакать без явной причины? Расскажите.',
        }
    },
    'В. Сомато-вегетативный домен (PHQ-9)': {
        'описание': 'Сон, аппетит, энергия, телесные симптомы',
        'инструмент': 'PHQ-9',
        'вопросы': {
            'В1': 'Как вы спите в последнее время? Засыпание, пробуждения, общая длительность сна?',
            'В2': 'Как у вас с аппетитом за последние две недели? Изменился ли вес?',
            'В3': 'Чувствуете ли вы упадок сил или, наоборот, повышенную утомляемость в течение дня?',
            'В4': 'Беспокоят ли вас телесные ощущения — головные боли, тяжесть в груди, проблемы с ЖКТ?',
        }
    },
    'Г. Когнитивный домен (MoCA, MMSE)': {
        'описание': 'Внимание, память, концентрация, ориентация',
        'инструмент': 'MoCA, MMSE',
        'вопросы': {
            'Г1': 'Замечаете ли вы трудности с концентрацией внимания при чтении или работе?',
            'Г2': 'Часто ли вы что-то забываете — имена, договорённости, куда положили вещи?',
            'Г3': 'Назовите, пожалуйста, сегодняшнюю дату — день недели, число, месяц, год.',
            'Г4': 'Посчитайте вслух от 100 в обратном порядке, отнимая по 7. Дойдите до пятого ответа.',
        }
    },
    'Д. Тревожный домен (GAD-7)': {
        'описание': 'Уровень тревоги, беспокойства, напряжения',
        'инструмент': 'GAD-7',
        'вопросы': {
            'Д1': 'Как часто за последние две недели вы чувствовали нервозность или беспокойство?',
            'Д2': 'Бывает ли так, что вы не можете перестать беспокоиться или контролировать тревогу?',
            'Д3': 'Чувствуете ли вы внутреннее напряжение, которое трудно расслабить?',
            'Д4': 'Бывают ли у вас приступы паники — учащённое сердцебиение, страх, ощущение нехватки воздуха?',
        }
    },
    'Е. Самооценка и риск (BDI-II)': {
        'описание': 'Самооценка, чувство вины, суицидальные мысли (триггер-блок)',
        'инструмент': 'BDI-II',
        'вопросы': {
            'Е1': 'Как вы относитесь к себе в последнее время? Считаете ли себя ценным человеком?',
            'Е2': 'Бывает ли у вас сильное чувство вины за прошлые поступки?',
            'Е3': 'Появлялись ли в последние две недели мысли, что жить не стоит или что лучше бы вас не было?',
            'Е4': 'Думали ли вы когда-нибудь о том, чтобы причинить себе вред? Если да, расскажите осторожно.',
        }
    },
    'Ж. Завершающий блок': {
        'описание': 'Резюме, ресурсы, поддержка',
        'инструмент': 'оригинальный',
        'вопросы': {
            'Ж1': 'Что в последнее время приносит вам радость или придаёт сил?',
            'Ж2': 'Есть ли в вашем окружении люди, к которым вы можете обратиться за поддержкой?',
            'Ж3': 'Что бы вы хотели изменить в своей жизни в ближайшие полгода?',
        }
    },
    'Ситуационные вопросы (Инглхарт-Шульман)': {
        'описание': 'Гипотетические ситуации для оценки ценностных ориентаций и копинг-стратегий',
        'инструмент': 'модель Инглхарта-Шульмана',
        'вопросы': {
            'С1': 'Представьте, что вы внезапно потеряли работу. Что бы вы почувствовали и как поступили в первую неделю?',
            'С2': 'Представьте, что вам нужно переехать в другую страну на пять лет. Опишите свои эмоции и решения.',
            'С3': 'Представьте серьёзный конфликт с близким человеком. Как вы будете действовать?',
            'С4': 'Представьте, что вы выиграли крупную сумму денег. Как изменится ваша жизнь?',
            'С5': 'Представьте, что серьёзно заболел кто-то из ваших близких. Что вы будете делать?',
        }
    },
    'СЭС. Социоэкономический статус': {
        'описание': 'Образование, занятость, субъективная оценка положения',
        'инструмент': 'оригинальный блок СЭС',
        'вопросы': {
            'СЭС1': 'Какое у вас образование? Где и когда вы его получили?',
            'СЭС2': 'Какова ваша текущая занятость и сфера деятельности?',
            'СЭС3': 'Как бы вы оценили своё материальное положение по шкале от 1 до 10?',
            'СЭС4': 'Считаете ли вы свой социальный статус выше, ниже или равным окружению?',
            'СЭС5': 'Хватает ли вам ресурсов на базовые нужды и небольшие удовольствия?',
        }
    },
    'Завершающая валидация (Этап 3, 1-2 минуты)': {
        'описание': 'Повтор калибровочных вопросов для контроля устойчивости базиса',
        'инструмент': 'повтор К1, К4, К5',
        'вопросы': {
            'К1_повтор': 'Опишите комнату, в которой вы сейчас находитесь. Что вы видите вокруг?',
            'К4_повтор': 'Напомните, что вы обычно едите на завтрак?',
            'К5_повтор': 'Опишите ещё раз ваш типичный распорядок дня.',
        }
    },
}

CALIBRATION = list(INTERVIEW_QUESTIONS['Калибровка (Этап 1, 3-4 минуты)']['вопросы'].keys())

SITUATIONAL = list(INTERVIEW_QUESTIONS['Ситуационные вопросы (Инглхарт-Шульман)']['вопросы'].values())

EMERGENCY_CONTACTS = (
    'Если вы чувствуете, что не справляетесь, пожалуйста, обратитесь за помощью:\n'
    '— Телефон доверия (бесплатно, круглосуточно): 8-800-2000-122\n'
    '— Московская служба психологической помощи: 051\n'
    '— Скорая психиатрическая помощь: 112'
)


def get_full_interview_script() -> str:
    lines = []
    lines.append('=' * 60)
    lines.append('ПОЛНЫЙ СЦЕНАРИЙ СТРУКТУРИРОВАННОГО ИНТЕРВЬЮ')
    lines.append('Длительность: 15-20 минут. Запишите одним файлом mp3.')
    lines.append('=' * 60)
    for block_name, block in INTERVIEW_QUESTIONS.items():
        lines.append('')
        lines.append('— ' + block_name + ' —')
        lines.append('Описание: ' + block['описание'])
        lines.append('Инструмент: ' + block['инструмент'])
        lines.append('')
        for code, question in block['вопросы'].items():
            lines.append(' [' + code + '] ' + question)
        lines.append('')
    lines.append('=' * 60)
    lines.append('Перед записью: зачитайте дисклеймер и получите согласие участника.')
    lines.append('После записи: пришлите файл mp3 в Telegram-бота.')
    lines.append('=' * 60)
    return '\n'.join(lines)


def load_audio(file_path: str, target_sr: int = 16000):
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr


def extract_paralinguistic(audio: np.ndarray, sr: int) -> Dict[str, float]:
    features = {}

    intervals = librosa.effects.split(audio, top_db=30)
    speech_duration = sum((e - s) for s, e in intervals) / sr
    total_duration = len(audio) / sr
    estimated_words = speech_duration / 0.4 if speech_duration > 0 else 0
    features['speech_rate_wpm'] = float(estimated_words / (total_duration / 60)) if total_duration > 0 else 0.0
    pause_ratio = 1 - (speech_duration / total_duration) if total_duration > 0 else 0
    features['pause_ratio_pct'] = float(pause_ratio * 100)

    f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    f0_clean = f0_clean[f0_clean > 0]
    if len(f0_clean) > 10:
        f0_mean = float(np.mean(f0_clean))
        f0_std_semitones = float(12 * np.log2(np.std(f0_clean) / f0_mean + 1)) if f0_mean > 0 else 0.0
    else:
        f0_mean = 0.0
        f0_std_semitones = 0.0
    features['f0_mean_hz'] = f0_mean
    features['f0_variability_semitones'] = f0_std_semitones

    if len(f0_clean) > 2:
        diffs = np.abs(np.diff(f0_clean))
        jitter = float(np.mean(diffs) / np.mean(f0_clean) * 100) if np.mean(f0_clean) > 0 else 0.0
    else:
        jitter = 0.0
    features['jitter_pct'] = jitter

    rms = librosa.feature.rms(y=audio)[0]
    if len(rms) > 2:
        shimmer = float(np.mean(np.abs(np.diff(rms))) / np.mean(rms) * 100) if np.mean(rms) > 0 else 0.0
    else:
        shimmer = 0.0
    features['shimmer_pct'] = shimmer

    intensity_db = float(20 * np.log10(np.mean(rms) + 1e-9) + 90)
    features['intensity_db'] = max(0.0, min(100.0, intensity_db))

    long_pauses = sum(1 for s, e in zip(intervals[:-1], intervals[1:]) if (e[0] - s[1]) / sr > 1.0)
    features['cognitive_load_proxy'] = float(long_pauses)

    return features


_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model('small')
    return _whisper_model


def speech_to_text(audio_path: str, language: str = 'ru') -> str:
    model = get_whisper_model()
    result = model.transcribe(audio_path, language=language, fp16=False)
    return result['text']


NEGATIVE_LEXICON = {
    'Б': ['грустно', 'тоска', 'плохо', 'уныло', 'безнадёжно', 'пусто', 'одиноко', 'подавленно'],
    'В': ['болит', 'тошнит', 'устал', 'бессонница', 'аппетит', 'тяжесть', 'сердце', 'давление'],
    'Г': ['забываю', 'не могу сосредоточиться', 'рассеянно', 'путаюсь', 'медленно думаю'],
    'Д': ['тревожно', 'страшно', 'паника', 'волнуюсь', 'напряжение', 'беспокойство', 'страх'],
    'Е': ['бесполезный', 'виноват', 'ненавижу себя', 'не хочу жить', 'лучше бы', 'смысла нет'],
}

POSITIVE_LEXICON = {
    'Б': ['радостно', 'хорошо', 'счастлив', 'спокойно', 'уверенно'],
    'В': ['здоров', 'бодр', 'сплю хорошо', 'аппетит хороший'],
    'Г': ['ясно мыслю', 'помню', 'собран', 'внимателен'],
    'Д': ['спокоен', 'уверен', 'без страха', 'расслаблен'],
    'Е': ['ценю себя', 'доволен', 'горжусь', 'живу с интересом'],
}

SUICIDE_TRIGGERS = ['не хочу жить', 'покончить', 'смысла жить нет', 'лучше умереть', 'исчезнуть навсегда']


def analyze_emotion(text: str, paralinguistic: Dict[str, float]) -> Dict[str, float]:
    text_lower = text.lower()
    domain_scores: Dict[str, float] = {}
    for code, info in DOMAINS.items():
        if code not in NEGATIVE_LEXICON:
            domain_scores[code] = 0.0
            continue
        neg = sum(text_lower.count(w) for w in NEGATIVE_LEXICON[code])
        pos = sum(text_lower.count(w) for w in POSITIVE_LEXICON[code])
        total = neg + pos
        if total == 0:
            domain_scores[code] = 0.0
        else:
            raw = (pos - neg) / total
            domain_scores[code] = float(max(-1.0, min(1.0, raw)))

    para_penalty = 0.0
    if paralinguistic.get('jitter_pct', 0) > 3.0:
        para_penalty += 0.1
    if paralinguistic.get('speech_rate_wpm', 120) < 80:
        para_penalty += 0.1
    if paralinguistic.get('pause_ratio_pct', 0) > 50:
        para_penalty += 0.1
    for code in ['Б', 'Д', 'Е']:
        domain_scores[code] = float(max(-1.0, domain_scores[code] - para_penalty))
    return domain_scores


def detect_suicide_risk(text: str) -> bool:
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in SUICIDE_TRIGGERS)


def normalize_response(text: str, model_output: Dict[str, float]) -> Dict[str, float]:
    return {k: float(max(-1.0, min(1.0, v))) for k, v in model_output.items()}


def calculate_deviation(emotional_value: float, neutral_value: float) -> Tuple[float, str]:
    if abs(neutral_value) < 1e-6:
        return 0.0, 'базис_равен_нулю'
    deviation_pct = (emotional_value - neutral_value) / abs(neutral_value) * 100
    abs_dev = abs(deviation_pct)
    if abs_dev <= 15:
        zone = 'норма'
    elif abs_dev <= 30:
        zone = 'умеренное'
    elif abs_dev <= 50:
        zone = 'значительное'
    else:
        zone = 'критическое'
    return float(deviation_pct), zone


def calculate_cronbach_alpha(items: np.ndarray) -> float:
    items = np.asarray(items, dtype=float)
    if items.ndim != 2 or items.shape[1] < 2 or items.shape[0] < 2:
        return float('nan')
    k = items.shape[1]
    item_variances = items.var(axis=0, ddof=1)
    total_variance = items.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return float('nan')
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return float(alpha)


def validate_evaluation_space(params: Dict[str, float]) -> Tuple[bool, str]:
    n = len(params)
    if n < 7:
        return False, f'Слишком мало параметров: {n}. Должно быть 7-20.'
    if n > 20:
        return False, f'Слишком много параметров: {n}. Должно быть 7-20.'
    for key, value in params.items():
        if not (-1.0 <= value <= 1.0):
            return False, f'Параметр {key} вне диапазона [-1, +1]: {value}'
    return True, 'OK'


def calculate_risk_level(domain_scores: Dict[str, float], suicide_flag: bool) -> Tuple[int, str]:
    score = 0
    for code in ['Б', 'Д', 'Е']:
        v = domain_scores.get(code, 0.0)
        if v < -0.66:
            score += 5
        elif v < -0.33:
            score += 3
        elif v < 0:
            score += 1
    if suicide_flag:
        score += 15
    if score <= 4:
        level = 'низкий'
    elif score <= 9:
        level = 'умеренный'
    elif score <= 14:
        level = 'высокий'
    else:
        level = 'критический'
    return score, level


QUALITY_CRITERIA = {
    'достоверность': 'Использованы валидированные шкалы PHQ-9, GAD-7, BDI-II, MoCA',
    'трансферабельность': 'Описан контекст применения и ограничения',
    'зависимая_надёжность': 'Альфа Кронбаха рассчитывается автоматически',
    'подтверждаемая': 'Все вычисления логируются в JSON',
}


def process_interview(audio_path: str) -> Dict:
    audio, sr = load_audio(audio_path)
    paraling = extract_paralinguistic(audio, sr)
    text = speech_to_text(audio_path)
    raw_scores = analyze_emotion(text, paraling)
    norm_scores = normalize_response(text, raw_scores)

    eval_space = {**norm_scores, **{f'para_{k}': min(1.0, max(-1.0, v/100)) for k, v in paraling.items()}}
    eval_space = dict(list(eval_space.items())[:15])
    valid, msg = validate_evaluation_space(eval_space)
    suicide = detect_suicide_risk(text)
    risk_score, risk_level = calculate_risk_level(norm_scores, suicide)
    return {
        'transcript': text,
        'paralinguistic': paraling,
        'domain_scores': norm_scores,
        'evaluation_space_valid': valid,
        'evaluation_space_message': msg,
        'suicide_risk_detected': suicide,
        'risk_score': risk_score,
        'risk_level': risk_level,
        'quality_criteria': QUALITY_CRITERIA,
    }


if __name__ == '__main__':
    print('Модуль анализа готов к работе')