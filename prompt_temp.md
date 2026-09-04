`H.4 — миграция Step 02 от task_key=(pos, primary_gloss_ru) к task_key=(pos, meaning_ru).`

Предлагаемое разбиение H.4 на 4 задачи одобряю. 

== H.4A — Новый task-input reader ==

Одобряю расположение файла в папке `src/sem_cat/io/`, но вместо `translation_tasks.py` возьми название `pos_meaning_ru_reader.py`.

== H.4B — Миграция task identity и TranslationTaskMetadata ==









== Тесты ==

Проверь в этом же файле `tests/sem_cat/test_reuse_analysis.py`:
1) какие ещё тесты следует переименовать для большей точности?
2) какие тесты перестали быть актуальными и их можно удалить?
3) какие тесты можно сделать более короткими или объединить с близкими?

Тест `test_no_reuse_meaningful_group_size_three_rows`, полагаю, можно удалить. 


 
То есть какие задачи нужно решить? Объедени эти задачи в группы для последовательного решения по каждой группе отдельно. 


Быстрые офлайн тесты показали следующие ошибки, что они показываю? Вот они:
```
(.venv) lunata@T34:/data/all/projects/git/dictorpus-space$ pytest tests/sem_cat -q
...
```

Сам Step 01 выдал в консоль результаты:
```
python3 -m src.sem_cat.01_reuse_analysis
Data dir: /data/all/projects/git/dictorpus-space/data/vepkar
Translate dir: /data/all/projects/git/dictorpus-space/data/sem_cat/2translate
Loading meanings...
Loaded 22110 rows for language 'vep'
Loaded 34298 rows for language 'olo'
Loaded 8362 rows for language 'lud'
Loaded 24241 rows for language 'krl'
Preparing meanings for analysis...
Total rows with non-empty primary gloss: 88984
Analyzing missing-English reuse by (pos, primary_gloss_ru)...
Writing reuse output files...
============================================================
Missing-English reuse analysis by (pos, primary_gloss_ru)
============================================================

Rows with non-empty primary_gloss_ru:        88984
Rows with existing human English:            17663
Rows missing English:                        71321

Among rows missing English:
  Reusable, one EN variant:                  2124
  Reusable, multiple EN variants:            364
  No reusable EN evidence:                   68833

Unique (pos, primary_gloss_ru) groups among rows missing English:
  Reusable, one EN variant:                  680
  Reusable, multiple EN variants:            90
  No reusable EN evidence:                   40872

Output directory: /data/all/projects/git/dictorpus-space/data/sem_cat/2translate
  - missing_en_reusable_unambiguous_pos_gloss_ru.csv
  - missing_en_reusable_ambiguous_pos_gloss_ru.csv
  - needs_translation_no_reuse.csv
  - missing_en_reusable_unambiguous_pos_gloss_ru_summary.csv
  - missing_en_reusable_ambiguous_pos_gloss_ru_summary.csv
```
