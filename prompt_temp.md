2translate:
`missing_en_reusable_unambiguous_by_task.csv`
`missing_en_reusable_ambiguous_by_task.csv`
`needs_translation_no_reuse.csv`
`missing_en_reusable_unambiguous_by_task_summary.csv`
`missing_en_reusable_ambiguous_by_task_summary.csv`


suggested_candidate_index = 1 в unambiguous output избыточен и семантически немного странен.
В unambiguous file и summary он всегда равен 1: например, PART::-ка с кандидатом just, now имеет suggested_candidate_index = 1. Это не неверно — первый и единственный кандидат действительно имеет индекс 1. Но поле было введено как UI-подсказка для выбора среди нескольких вариантов, то есть прежде всего для ambiguous rows. Варианты исправления:
1) Предпочтительный: оставить столбец в обоих schema для единообразия, но в unambiguous файлах писать пустое значение. Там выбирать нечего.
2) Альтернатива: оставить 1, но чётко документировать: “для unambiguous это обозначает единственный кандидат, а не рекомендацию”.

Я выбираю вариант: убрать suggested_candidate_index там, где он избыточен.
