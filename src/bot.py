import os
import asyncio
import tempfile
import logging
from pathlib import Path

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import CommandStart, Command

from analysis_core import process_interview, EMERGENCY_CONTACTS, DOMAINS, INTERVIEW_QUESTIONS, get_full_interview_script

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('bot')

load_dotenv()  # загружаем переменные из .env
BOT_TOKEN = os.getenv('BOT_TOKEN')
if not BOT_TOKEN:
    raise RuntimeError('BOT_TOKEN не найден. Создайте файл .env с токеном.')

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

DISCLAIMER = (
    'Внимание. Этот бот — скрининговый прототип в рамках дипломной работы.\n'
    'Он не ставит диагнозов и не заменяет консультацию специалиста.\n'
    'Отправляя аудио, вы подтверждаете информированное согласие на анализ.\n'
    'Аудио удаляется сразу после обработки.'
)

# ============================================================
# СЛОВЕСНЫЕ РАСШИФРОВКИ ДОМЕНОВ
# ============================================================
DOMAIN_MEANING = {
    'А': {
        'positive': 'факторологические данные стабильны, биографический фон без выраженных стресс-факторов.',
        'neutral': 'факторологический фон нейтрален, значимые жизненные события не выявлены.',
        'negative': 'в анамнезе присутствуют значимые стресс-факторы или жизненные события, требующие внимания.'
    },
    'Б': {
        'positive': 'эмоциональный фон позитивный, признаки сниженного настроения не выражены.',
        'neutral': 'эмоциональный фон сбалансирован, выраженных признаков депрессии или эйфории нет.',
        'negative': 'выявлены признаки сниженного настроения, ангедонии или эмоциональной подавленности (ориентир — PHQ-9, BDI-II).'
    },
    'В': {
        'positive': 'сомато-вегетативная сфера в норме: сон, аппетит и энергия не нарушены.',
        'neutral': 'значимых жалоб на сон, аппетит или телесные симптомы не выявлено.',
        'negative': 'выявлены жалобы сомато-вегетативного характера: нарушения сна, аппетита, утомляемость или телесные ощущения.'
    },
    'Г': {
        'positive': 'когнитивные функции сохранны: внимание, память и концентрация в норме.',
        'neutral': 'когнитивная сфера без выраженных особенностей.',
        'negative': 'возможны трудности с концентрацией, памятью или скоростью мышления (ориентир — MoCA, MMSE).'
    },
    'Д': {
        'positive': 'уровень тревоги низкий, признаков беспокойства не выявлено.',
        'neutral': 'тревожный фон умеренный, в пределах ситуативной нормы.',
        'negative': 'выявлены признаки повышенной тревожности, беспокойства или соматического напряжения (ориентир — GAD-7).'
    },
    'Е': {
        'positive': 'самооценка устойчивая, признаков самообесценивания или суицидального риска не выявлено.',
        'neutral': 'самооценка стабильна, выраженных деструктивных установок нет.',
        'negative': 'присутствуют признаки сниженной самооценки, чувства вины или суицидальной идеации — требуется внимание специалиста.'
    },
    'Ж': {
        'positive': 'есть ресурсы поддержки и позитивные жизненные ориентиры.',
        'neutral': 'завершающий блок без выраженных особенностей.',
        'negative': 'ограничены социальные ресурсы или отсутствуют значимые источники поддержки.'
    },
}


def interpret_domain_score(code: str, score: float) -> str:
    # Преобразует числовую оценку домена в словесное описание со степенью выраженности
    meanings = DOMAIN_MEANING.get(code, {})
    if score >= 0.66:
        level = 'выраженная позитивная тенденция'
        text = meanings.get('positive', '')
    elif score >= 0.33:
        level = 'умеренная позитивная тенденция'
        text = meanings.get('positive', '')
    elif score > -0.33:
        level = 'нейтральный баланс'
        text = meanings.get('neutral', '')
    elif score > -0.66:
        level = 'умеренная негативная тенденция'
        text = meanings.get('negative', '')
    else:
        level = 'выраженная негативная тенденция'
        text = meanings.get('negative', '')
    return f'{level}. {text}'


# ============================================================
# ИНТЕРПРЕТАЦИЯ ПАРАЛИНГВИСТИКИ
# ============================================================
# Каждый параметр: (минимум_нормы, максимум_нормы, текст_ниже_нормы, текст_выше_нормы, текст_в_норме)
PARA_NORMS = {
    'speech_rate_wpm': {
        'name': 'Темп речи',
        'unit': 'слов/мин',
        'min': 80, 'max': 200,
        'low': 'замедленный темп речи. Может указывать на сниженное настроение, утомление или когнитивную загруженность.',
        'high': 'ускоренный темп речи. Может указывать на тревогу, возбуждение или эмоциональное напряжение.',
        'ok': 'темп речи в пределах нормы.'
    },
    'pause_ratio_pct': {
        'name': 'Доля пауз',
        'unit': '%',
        'min': 0, 'max': 50,
        'low': 'очень мало пауз — речь плотная, без передышек.',
        'high': 'повышенная доля пауз. Может говорить о затруднении формулировок, эмоциональном торможении или раздумьях.',
        'ok': 'паузы в пределах нормы, ритм речи естественный.'
    },
    'f0_variability_semitones': {
        'name': 'Вариативность тона',
        'unit': 'полутонов',
        'min': 1, 'max': 10,
        'low': 'монотонная речь с низкой вариативностью тона. Часто связана со сниженным эмоциональным фоном.',
        'high': 'очень высокая вариативность тона — речь эмоционально насыщенная, возможна экспрессивность или возбуждение.',
        'ok': 'интонационная окраска в пределах нормы, эмоциональная экспрессия естественная.'
    },
    'jitter_pct': {
        'name': 'Jitter (дрожание частоты)',
        'unit': '%',
        'min': 0, 'max': 5,
        'low': 'дрожание частоты минимальное — голос стабильный.',
        'high': 'повышенное дрожание частоты голоса. Может указывать на тревогу, усталость или соматическое напряжение.',
        'ok': 'дрожание частоты в норме, голос стабилен.'
    },
    'shimmer_pct': {
        'name': 'Shimmer (дрожание амплитуды)',
        'unit': '%',
        'min': 0, 'max': 10,
        'low': 'амплитуда голоса стабильна, без колебаний громкости.',
        'high': 'повышенное дрожание амплитуды. Часто наблюдается при эмоциональном напряжении или утомлении голосового аппарата.',
        'ok': 'амплитуда голоса стабильна, в пределах нормы.'
    },
    'intensity_db': {
        'name': 'Интенсивность голоса',
        'unit': 'dB',
        'min': 30, 'max': 90,
        'low': 'тихий голос. Может указывать на сниженный эмоциональный тонус или дискомфорт в коммуникации.',
        'high': 'очень громкий голос. Возможны эмоциональное возбуждение или попытка компенсации.',
        'ok': 'громкость голоса в пределах нормы.'
    },
}


def interpret_paralinguistic_param(key: str, value: float) -> str:
    cfg = PARA_NORMS.get(key)
    if cfg is None:
        return ''
    if value < cfg['min']:
        status = 'НИЖЕ НОРМЫ'
        text = cfg['low']
    elif value > cfg['max']:
        status = 'ВЫШЕ НОРМЫ'
        text = cfg['high']
    else:
        status = 'НОРМА'
        text = cfg['ok']
    return f'[{status}] {text}'


# ============================================================
# ОБЩАЯ ИНТЕРПРЕТАЦИЯ И РЕКОМЕНДАЦИИ
# ============================================================
def build_overall_summary(result: dict) -> str:
    scores = result['domain_scores']
    risk_level = result['risk_level']
    suicide = result['suicide_risk_detected']

    # Находим самые проблемные и самые сильные домены
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    weakest = [(c, s) for c, s in sorted_scores if s < -0.2][:3]
    strongest = [(c, s) for c, s in sorted(scores.items(), key=lambda x: -x[1]) if s > 0.2][:2]
    lines = ['ОБЩАЯ ИНТЕРПРЕТАЦИЯ', '']
    if suicide:
        lines.append('ВНИМАНИЕ: в речи выявлены маркеры суицидального риска. Это приоритетный сигнал, требующий немедленного внимания специалиста.')
        lines.append('')
    if weakest:
        lines.append('Зоны, требующие внимания:')
        for code, score in weakest:
            name = DOMAINS[code]['name']
            lines.append(f'  — {code} ({name}): {score:+.2f} — {interpret_domain_score(code, score)}')
        lines.append('')
    else:
        lines.append('Выраженных негативных тенденций по доменам не выявлено.')
        lines.append('')
    if strongest:
        lines.append('Ресурсные зоны:')
        for code, score in strongest:
            name = DOMAINS[code]['name']
            lines.append(f'  — {code} ({name}): {score:+.2f}')
        lines.append('')
    # Уровень риска
    risk_descriptions = {
        'низкий': 'Низкий уровень риска. Скрининг не выявил значимых отклонений. Рекомендуется поддерживающее наблюдение.',
        'умеренный': 'Умеренный уровень риска. Есть отдельные признаки, заслуживающие внимания. Рекомендуется консультация специалиста для уточнения.',
        'высокий': 'Высокий уровень риска. Скрининг выявил выраженные негативные тенденции. Настоятельно рекомендуется обратиться к психологу или психотерапевту.',
        'критический': 'Критический уровень риска. Совокупность признаков требует немедленного обращения к специалисту. См. контакты экстренной помощи ниже.'
    }
    lines.append('Интерпретация уровня риска:')
    lines.append(risk_descriptions.get(risk_level, ''))
    return '\n'.join(lines)


# ============================================================
# ОБРАБОТЧИКИ КОМАНД
# ============================================================
@dp.message(CommandStart())
async def on_start(message: Message):
    await message.answer(
        'Здравствуйте. Я прототип системы психоэмоционального профилирования.\n\n'
        + DISCLAIMER + '\n\n'
        'Доступные команды:\n'
        '/questions — показать все вопросы интервью по блокам\n'
        '/script — получить сценарий одним текстовым файлом\n'
        '/help — структура и длительность сессии\n\n'
        'После записи пришлите аудиофайл mp3 или голосовое сообщение, '
        'и я верну анализ по шести доменам в шкале от -1.0 до +1.0 с расшифровкой каждого показателя.'
    )


@dp.message(Command('help'))
async def on_help(message: Message):
    await message.answer(
        'Структура интервью:\n'
        'Этап 1 (3-4 мин) — калибровка К1-К7.\n'
        'Этап 2 (10-15 мин) — основные домены А-Ж.\n'
        'Этап 3 (1-2 мин) — повтор К1, К4, К5.\n\n'
        'Запишите интервью одним файлом mp3 и пришлите сюда.'
    )


@dp.message(Command('questions'))
async def on_questions(message: Message):
    await message.answer('Полный сценарий интервью. Запишите ответы одним файлом mp3.')
    for block_name, block in INTERVIEW_QUESTIONS.items():
        text_lines = ['— ' + block_name + ' —']
        text_lines.append('Описание: ' + block['описание'])
        text_lines.append('Инструмент: ' + block['инструмент'])
        text_lines.append('')
        for code, question in block['вопросы'].items():
            text_lines.append('[' + code + '] ' + question)
        await message.answer('\n'.join(text_lines))
    await message.answer('Готово. После записи пришлите mp3 или голосовое сообщение для анализа.')


@dp.message(Command('script'))
async def on_script(message: Message):
    script = get_full_interview_script()
    file = BufferedInputFile(script.encode('utf-8'), filename='interview_script.txt')
    await message.answer_document(file, caption='Полный сценарий интервью для распечатки')


async def _download_to_temp(message: Message) -> Path:
    if message.audio:
        file_id = message.audio.file_id
        suffix = '.mp3'
    elif message.voice:
        file_id = message.voice.file_id
        suffix = '.ogg'
    elif message.document and message.document.mime_type and 'audio' in message.document.mime_type:
        file_id = message.document.file_id
        suffix = '.mp3'
    else:
        return None
    file = await bot.get_file(file_id)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir='temp')
    tmp.close()
    await bot.download_file(file.file_path, tmp.name)
    return Path(tmp.name)


def _format_report(result: dict) -> str:
    lines = ['РЕЗУЛЬТАТЫ АНАЛИЗА ИНТЕРВЬЮ', '=' * 40, '']

    # 1. Оценки по доменам со словесной расшифровкой
    lines.append('ОЦЕНКИ ПО ДОМЕНАМ (шкала от -1.0 до +1.0):')
    lines.append('')
    for code, info in DOMAINS.items():
        score = result['domain_scores'].get(code, 0.0)
        lines.append(f'{code} — {info["name"]}: {score:+.2f}  (источник: {info["source"]})')
        lines.append(f'   Интерпретация: {interpret_domain_score(code, score)}')
        lines.append('')

    # 2. Паралингвистика с расшифровкой каждого показателя
    lines.append('=' * 40)
    lines.append('ПАРАЛИНГВИСТИЧЕСКИЕ ПОКАЗАТЕЛИ:')
    lines.append('')
    p = result['paralinguistic']
    para_order = [
        ('speech_rate_wpm', p.get('speech_rate_wpm', 0), '{:.0f}'),
        ('pause_ratio_pct', p.get('pause_ratio_pct', 0), '{:.1f}'),
        ('f0_variability_semitones', p.get('f0_variability_semitones', 0), '{:.2f}'),
        ('jitter_pct', p.get('jitter_pct', 0), '{:.2f}'),
        ('shimmer_pct', p.get('shimmer_pct', 0), '{:.2f}'),
        ('intensity_db', p.get('intensity_db', 0), '{:.1f}'),
    ]
    for key, value, fmt in para_order:
        cfg = PARA_NORMS.get(key, {})
        name = cfg.get('name', key)
        unit = cfg.get('unit', '')
        norm_min = cfg.get('min', '?')
        norm_max = cfg.get('max', '?')
        lines.append(f'{name}: {fmt.format(value)} {unit}  (норма {norm_min}-{norm_max})')
        lines.append(f'   {interpret_paralinguistic_param(key, value)}')
        lines.append('')

    # 3. Уровень риска и валидность
    lines.append('=' * 40)
    lines.append(f'Уровень риска: {result["risk_level"]} (балл {result["risk_score"]})')
    lines.append(f'Валидность пространства оценки: {result["evaluation_space_message"]}')
    lines.append('')

    # 4. Общая интерпретация и рекомендации
    lines.append('=' * 40)
    lines.append(build_overall_summary(result))
    lines.append('')

    # 5. Экстренные контакты при критическом риске
    if result['suicide_risk_detected'] or result['risk_level'] == 'критический':
        lines.append('=' * 40)
        lines.append(EMERGENCY_CONTACTS)
        lines.append('')
    lines.append('=' * 40)
    lines.append(DISCLAIMER)
    return '\n'.join(lines)


@dp.message(F.audio | F.voice | F.document)
async def on_audio(message: Message):
    await message.answer('Файл получен. Идёт анализ, это занимает 1-3 минуты…')
    audio_path = None
    try:
        audio_path = await _download_to_temp(message)
        if audio_path is None:
            await message.answer('Не удалось распознать аудио. Пришлите mp3 или голосовое.')
            return
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, process_interview, str(audio_path))
        report = _format_report(result)
        # Telegram режет сообщения длиннее 4096 символов — отправляем по частям
        chunk_size = 3500
        for i in range(0, len(report), chunk_size):
            await message.answer(report[i:i + chunk_size])
    except Exception as e:
        log.exception('Ошибка обработки')
        await message.answer(f'Произошла ошибка: {e}')
    finally:
        if audio_path and audio_path.exists():
            try:
                audio_path.unlink()
            except Exception:
                pass


async def main():
    Path('temp').mkdir(exist_ok=True)
    print('Бот запущен')
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())