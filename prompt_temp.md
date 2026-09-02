2translate:
`missing_en_reusable_unambiguous_by_task.csv`
`missing_en_reusable_ambiguous_by_task.csv`
`needs_translation_no_reuse.csv`
`missing_en_reusable_unambiguous_by_task_summary.csv`
`missing_en_reusable_ambiguous_by_task_summary.csv`


Обсудим файл первого шага: `needs_translation_no_reuse.csv`.
Я предлагаю на шаге 2 и следующих работать не со всеми 5 файлами в папке `data/sem_cat/2translate`, а только с теми строками/задачами, которые вовсе не имеют перевод на английский язык, то есть я говорю про информацию в файле `needs_translation_no_reuse.csv`. 

Вот фрагмент этого файла `needs_translation_no_reuse.csv`:
```
id,meaning_id,lemma_id,lemma,lang,task_pos,meaning_ru,primary_gloss_ru,concept_id,category_id,pos_gloss_ru_key,missing_row_count_for_pos_gloss_ru
11838,71523,60871,muraškkeitoz,vep,NOUN,морошковое варенье,морошковое варенье,,,NOUN::морошковое варенье,2
16812,66314,57251,muur’oivaren’n’u,olo,NOUN,морошковое варенье,морошковое варенье,,,NOUN::морошковое варенье,2
11839,71524,60872,murašklehtez,vep,NOUN,лист морошки,лист морошки,,,NOUN::лист морошки,1
11840,71525,60873,muraškmänd,vep,NOUN,"сухой участок болота, поросший морошкой","сухой участок болота, поросший морошкой",,,"NOUN::сухой участок болота, поросший морошкой",1
11841,71526,60874,muraškpaik,vep,NOUN,"место, обильно поросшее морошкой","место, обильно поросшее морошкой",,,"NOUN::место, обильно поросшее морошкой",2
11842,71527,60875,murašksija,vep,NOUN,"место, обильно поросшее морошкой","место, обильно поросшее морошкой",,,"NOUN::место, обильно поросшее морошкой",2
```

Из этого фрагмента видно, что нужно переводить следующие русские уникальные в этом файле фразы или слова:
1) морошковое варенье
2) лист морошки
3) "сухой участок болота, поросший морошкой"
4) "место, обильно поросшее морошкой"

То есть на первом шаге вместе с создание файла `needs_translation_no_reuse.csv` нужно создать файл, в котором будут содержаться уникальные фразы на русском, без повторов, которые можно переводить на английский. То есть аналог файлов `missing_en_reusable_unambiguous_by_task_summary.csv`, `missing_en_reusable_ambiguous_by_task_summary.csv`, но без каких-либо переводов на английский. Предложи название этого файла и давай тщательно обсудим, какие именно столбцы он должен содержать. 

И следующий тесно связанный вопрос, точнее два варианта подхода к переводу:
1) переводить отдельно каждое из четырёх примеров выше, связанных с морошкой. 
2) объединить слова и фразы, имеющие общие ключевые слова (например 4 фразы, связанные с морошкой), и предлагать LLM и переводчикам группы слов близкие по смыслу для перевода. 

Из используемых моделей, какие модели позволяют переводить группы слов, например, соединённые (конкатенация) точкой с пробелом, то есть как один небольшой текст из предложений?
Повысит ли это точность перевода, так как будет задавать больший общий контекст, чем, например, одно короткое слово, особенно если оно многозначное?





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

В файле `needs_translation_no_reuse.csv` вот первая строка и несколько строк:
```
id,meaning_id,lemma_id,lemma,lang,task_pos,meaning_ru,primary_gloss_ru,concept_id,category_id,pos_gloss_ru_key,missing_row_count_for_pos_gloss_ru
1684,27910,24423,boljom,vep,NOUN,брусничный напиток,брусничный напиток,,,NOUN::брусничный напиток,3
1698,39759,34110,bolvezi,vep,NOUN,брусничный напиток,брусничный напиток,,,NOUN::брусничный напиток,3
2261,61455,53823,buoluvezi,olo,NOUN,брусничный напиток,брусничный напиток,,,NOUN::брусничный напиток,3
1685,39750,34101,bolkeitoz,vep,NOUN,брусничное варенье,брусничное варенье,,,NOUN::брусничное варенье,2
2260,61454,53822,buoluvaren’n’u,olo,NOUN,брусничное варенье,брусничное варенье,,,NOUN::брусничное варенье,2
```
