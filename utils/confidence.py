import config

def evaluate_confidence(similarity_score: float) -> dict:
    """
    Оценивает уверенность модели на основе оценки сходства.
    Возвращает словарь с уровнем уверенности и флагом для эксперта.
    """
    if similarity_score >= config.SIMILARITY_THRESHOLD_HIGH:
        return {
            "level": "Высокая",
            "needs_expert": False,
            "message": "Модель уверена в результате. Гипотеза надежна."
        }
    elif similarity_score >= config.SIMILARITY_THRESHOLD_LOW:
        return {
            "level": "Средняя",
            "needs_expert": False,
            "message": "Результат вероятен, но рекомендуется сверить с каталогом."
        }
    else:
        return {
            "level": "Низкая",
            "needs_expert": True,
            "message": "⚠️ ВНИМАНИЕ: Требуется обязательная экспертная проверка!"
        }