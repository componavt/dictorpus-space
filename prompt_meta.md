После применения предложенного тобой промпта получен **актуальный repository snapshot** в прилагаемом файле `semcat_NN.md`.

**Что я лично проверил в приложении после предыдущей итерации:**
<конкретные наблюдения: что работает, что не работает, сообщения об ошибках, особенности интерфейса>

**Новые пожелания или приоритеты:**
<если есть; иначе: нет>

Вот output в VS Code после реализации агентом ИИ предложенного тобой промпта:
```
```

Ответь на основе актуального repository snapshot и моих наблюдений выше:
- Выполнена ли поставленная в предыдущем промпте задача/этап и что нужно доработать?
- Тщательно проанализируй и детально сообщи, какие ошибки есть в коде и как их можно исправить?

* * * 

Твоя задача: на основе того, что нужно изменить в коде (твой подробный текст выше и мои небольшие замечания), сформулируй промпт для CLI ИИ-чата для изменений кода репозитория (детали по коду см. в файле `semcat_35_broken2.txt`).

Важный контекст:
- Ранее был приложен один файл: `semcat_35_broken2.txt`.
- Считай этот файл repository snapshot.
- Этот repository snapshot дан только тебе для анализа в этом чате.
- В итоговом prompt for CLI AI-chat нельзя советовать CLI AI-chat читать этот файл или ссылаться на него, потому что CLI AI-chat в своей среде и так имеет прямой доступ к реальным файлам проекта.
- Поэтому итоговый английский prompt must talk about real project files, real modifications, and real implementation strategy.

Для важных и критических мест пиши подробные фрагменты кода в этом промпте. Кроме промпта для CLI AI-chat больше ничего не пиши.

Прикажи CLI AI-chat, чтобы при указании файлов всегда обращался к ним по полному имени файла, поскольку в противном случае агент KiloCode будет вызывать внутренний “инструмент редактирования” с ошибочными аргументами, то есть без обязательного поля filePath.

* * * 



* * * 

Успешно выполнен промпт для CLI AI-chat и внесены изменения в файлы. Этот чат после работы выдал следующий summary по своей работе в VS Code:
```
some text...
```

Прилагаю итоговый репозиторий, собранный в один файл `semcat_34_broken.txt` с помощью gitingest.

Итак, мне нужен детальный и подробный промпт (с примерами кода для критических участков) для CLI AI-chat (Deepseek/Qwen) на английском для исправления ошибок в файлах репозитория, собранных в один файл `digest_git_ingest_22_Tower_empty_translations.txt` с помощью gitingest. Этот промпт должен исправлять перечисленные тобой выше ошибки в коде и в тестах с учётом моих замечаний.

Уточняю данные относительно файла `semcat_27_back_to_translation.txt`:
- Считай этот файл repository snapshot.
- Этот repository snapshot дан только тебе для анализа в этом чате.
- В итоговом prompt for CLI AI-chat нельзя советовать CLI AI-chat читать этот файл или ссылаться на него, потому что CLI AI-chat в своей среде и так имеет прямой доступ к реальным файлам проекта.

Выдай только промпт, ничего кроме промпта не пиши.




`semcat_31_not_finished.txt` - прилагаю этот файл (repository snapshot), который содержит обновлённые файлы репозитория, собранные в один файл с помощью команды gitingest.

Проблема в том, что CLI AI-chat не завершил работу, поэтому, возможно, некоторые файлы находятся в состоянии не совсем consistency. Проверь, какие видишь ошибки в коде? Что нужно доделать в коде?

Пиши только текст без кода. Нумеруй свои положения. Подробно и обстоятельно объясни, почему эти ошибки нужно исправить и как именно их нужно исправить.



--- 
Выдай только промпт, ничего кроме промпта не пиши.
---
Вот Summary сообщение от CLI-чат в VS Code после внесения исправлений в код:
---


Предложения:

1) Одобряю. 

Но функция rename_for_internal_use() - лишняя, поскольку ты стабильно ошибаешься: "текущие extraction- и metadata-функции работают по колонкам вроде meaningru". На самом деле, в коде эти колонки типа meaning_ru через подчёркивание. 

2) Одобряю. 

"корректно ли считать строку со значением meaning_en=" " именно needs_mt-строкой и одновременно печатать её полный row dump в консоль как проблемную?" - Да, корректно. Печатать в консоль не нужно. 

3) Одобряю. 

"Деталь реализации: сводка должна печататься по каждому meanings_krl.csv, meanings_lud.csv, meanings_olo.csv, meanings_vep.csv, а отдельно — два счётчика по паре (pos, primary_gloss_ru): reusable_unambiguous для случаев с ровно одним уже существующим English вариантом и reusable_ambiguous для случаев с несколькими разными English вариантами при наличии ещё непереведённых строк." - да. Дополнительно сводка по всем четырём файлам. 

"Уточняющий вопрос: для ambiguous-cases тебе достаточно только количества и первых N примеров в консоли, или всё-таки нужен ещё отдельный вспомогательный CSV, даже если основной отчёт остаётся только консольным?" Нужен отдельный вспомогательный CSV.

4) Одобряю. Подтверждаю, что pos надо сделать частью ключа обязательно, а при отсутствии POS использовать UNKNOWN.

5) Да, можешь забыть про “корзину состояний”. 

У шага 02 уже есть --offset и --limit, поэтому не нужно никаких --remaining-limit или --todo-limit.

6) Одобряю. Пусть lang был единственным кратким идентификатором источника. 

7) Одобряю.


Прилагаю итоговый репозиторий, собранный в один файл `digest_git_ingest_26.txt` с помощью gitingest.

Итак, мне нужен детальный и подробный промпт (с примерами кода для критических участков) для CLI AI-chat (Deepseek/Qwen) на английском для внесения изменений в файлы репозитория, собранных в один файл `digest_git_ingest_26.txt` с помощью gitingest. Этот промпт должен вносить указанные тобой выше предложения в код.

Уточняю данные относительно файла `digest_git_ingest_26.txt`:
- Считай этот файл repository snapshot.
- Этот repository snapshot дан только тебе для анализа в этом чате.
- В итоговом prompt for CLI AI-chat нельзя советовать CLI AI-chat читать этот файл или ссылаться на него, потому что CLI AI-chat в своей среде и так имеет прямой доступ к реальным файлам проекта.

Выдай только промпт, ничего кроме промпта не пиши.



======================



Сформулируй CLI prompt (на основе твоего предыдущего ответа и repository snapshot в файле `semcat_NN.md`) только для подзадачи <название выбранной подзадачи>.

CLI prompt для Qwen3-Coder-Next должен:
   - быть на английском языке;
   - описывать реальные изменения реального текущего репозитория;
   - не упоминать repository snapshot и не советовать читать `semcat_NN.md`;
   - всегда указывать полный относительный путь при упоминании файла, например `lib/game_page.dart`, а не `game_page.dart`;
   - включать фрагменты кода только для критической логики, сложных частей кода;
   - указывать краткий ожидаемый итог подзадачи в наблюдаемой форме;
   - требовать тест для исправления, если проблема относится к воспроизводимой логике, данным, сохранению состояния;
   - не считать задачу выполненной только по наличию строки, метода, класса, импорта, grep-вывода или успешной компиляции;
   - не скрывать ошибки через изменение `analysis_options.yaml`, `// ignore`, `// ignore_for_file` или lint suppression; если исключение действительно необходимо, агент обязан явно объяснить причину;
   - не менять несвязанные файлы, зависимости, без явного объяснения необходимости;
   - требовать в финальном отчёте: изменённые файлы, реализованные требования, выполненные тесты, известные ограничения.

Включи следующий блок в итоговый промпт:
```
============================================================
TOOL-CALL DISCIPLINE
============================================================

- Use the editor's native structured tool interface; do not print pseudo-tool calls,
  XML tool calls, Markdown JSON blocks, or narration instead of invoking a tool.
- For each edit, make one small atomic replacement in one full-path file.
- Before editing, read the exact target fragment from the file.
- The edit must send all required fields in the tool schema:
  filePath, oldString, newString.
- Use camelCase schema names exactly; do not use file_path, old_string, or new_string.
- If a tool validation error occurs, retry once with the exact required schema.
  Do not repeat narration or issue another empty tool call.
- Prefer several small edits over one large write containing a whole long source file.
```

===============================

Formulate one CLI prompt for Qwen3-Coder-Next based on the immediately preceding analysis and the current repository state represented in the attached repository snapshot file `semcat_NN.md`.

The CLI prompt must cover only the selected subtask:

<SELECTED SUBTASK NAME>

The output must be the CLI prompt only, in English. Do not add a Russian introduction, explanation, or commentary outside the prompt.

============================================================
PROMPT PURPOSE AND SCOPE
============================================================

Write a precise implementation prompt for a real Python repository. The repository contains a VepKar semantic-categorization pipeline, including:

- Python CLI modules under `src/sem_cat/`;
- pandas DataFrame transformations and CSV files;
- Step 01 missing-English reuse analysis;
- Step 02 LLM/machine-translation workflow;
- Step 03 comparison of model translations;
- offline pytest tests under `tests/sem_cat/`;
- optional HuggingFace / PyTorch / NLTK components which may be unavailable in the local environment.

The prompt must direct the coding agent to modify real project files and implement the selected subtask only.

Do not tell the coding agent to read, inspect, or refer to `semcat_NN.md`, any repository snapshot, this conversation, or any attached file. The coding agent already has direct access to the real repository.

Do not broaden the selected subtask into adjacent planned work. If the current code contains known out-of-scope failures, preserve them and report them clearly rather than “fixing” them opportunistically.

============================================================
PATH AND FILE-NAMING DISCIPLINE
============================================================

- Every mention of a repository file must use its full repository-relative path.
  Correct: `src/sem_cat/pipeline/reuse_analysis.py`
  Incorrect: `reuse_analysis.py`

- Use the repository’s real current package/module paths and filenames.
  Do not invent paths, normalize names speculatively, or rename modules solely for style.

- Before asking for a modification, identify the exact in-scope files.
  Include only files genuinely needed for the selected subtask.

- Do not modify unrelated files, dependencies, package versions, lock files, project configuration,
  lint configuration, README files, or test settings unless the selected subtask directly requires it.
  If any additional file is necessary, require the agent to explain the dependency in its final report.

============================================================
PYTHON / DATA PIPELINE IMPLEMENTATION RULES
============================================================

The coding agent must treat CSV schemas and DataFrame column values as public data contracts.

For every changed CSV-producing or CSV-consuming behavior, the prompt must explicitly state:

1. The exact output file name(s).
2. Whether each file is always written or written only when non-empty.
3. The expected row semantics: which source rows must be included and excluded.
4. The exact required columns and their stable order, when schema changes are part of the subtask.
5. The intended values for derived fields, including empty/null/zero behavior.
6. Whether a field is:
   - meaningful,
   - a fixed invariant,
   - a temporary internal field,
   - or intentionally absent from the output.

Do not permit “schema correctness” to be satisfied merely because missing columns were appended as empty strings by a helper such as `ensure_columns(...)`.

Require that derived values be computed at the point where all needed group-level information is available. For example:

- group statistics must be calculated inside the grouping/classification loop;
- summary-only provenance must be derived from the relevant source subsets;
- writers should serialize already-correct DataFrames rather than reconstruct business logic;
- output writers must not silently fabricate values that should have been computed earlier.

When the selected subtask changes a mutually exclusive classification, require a conservation invariant. For example:

```python
assert total_missing_rows == (
    unambiguous_reuse_rows
    + ambiguous_reuse_rows
    + no_reuse_rows
)
```

Use the repository’s actual field names, but require the equivalent semantic invariant whenever applicable.

Preserve the established distinction between:

- raw source fields such as `pos`;
- normalized pipeline fields such as `task_pos`;
- Step-01 review naming such as `pos_gloss_ru_key`;
- Step-02/Step-03 internal identity such as `task_key`.

Do not perform broad terminology renames unless the selected subtask explicitly requires them.

============================================================
CODE EXAMPLES AND IMPLEMENTATION GUIDANCE
============================================================

Include code fragments whenever they materially reduce ambiguity or prevent a known class of bug.

Do not restrict snippets to only tiny fragments. For critical pandas/grouping/CSV logic, provide enough local context for correct implementation, normally including:

- function signature;
- relevant constants/schema lists;
- the relevant branch or loop;
- expected input and output DataFrame columns;
- the required ordering of computation;
- important assertions or test examples.

Prefer a compact but complete local implementation pattern of roughly 15–60 lines when needed, rather than an isolated 2-line patch that hides the data flow.

For example, if a no-reuse group must carry a group size, show the complete intended branch:

```python
if candidate_count == 0:
    no_reuse_rows = missing_group.copy()
    no_reuse_rows["missing_row_count_for_pos_gloss_ru"] = len(missing_group)
    groups.append(
        {
            "kind": "no_reuse",
            "rows": no_reuse_rows,
        }
    )
    continue
```

Then state explicitly which fields must not be added because they would be fixed tautologies for that output class.

Do not paste whole unrelated source files. Do not prescribe exact code where repository-local conventions need inspection first. Use code snippets for critical logic, not as a substitute for understanding the existing code.

============================================================
TESTING REQUIREMENTS
============================================================

If the selected subtask changes reproducible logic, DataFrame transformation, CSV output, schema, persistence, classification, or CLI behavior, require focused tests.

Tests must:

- live under the real relevant full path, normally `tests/sem_cat/...`;
- be offline, deterministic, and fast;
- use small synthetic pandas DataFrames and `tmp_path` for files;
- not require real VepKar CSV exports;
- not require an LLM backend, HuggingFace download, GPU, CUDA, NLTK download, proxy access, or network access;
- assert behavior and output content, not only that a function exists.

For each bug fixed, require a regression test that would have failed before the change.

For CSV-writing tasks, require tests for both:

1. a non-empty output with representative values; and
2. an empty output with stable headers and zero data rows, if the output is intended to exist even when empty.

Require tests to check:
- exact output file existence;
- rows included and excluded;
- exact required columns and stable order where relevant;
- values of important computed fields;
- absence of columns which must not be public;
- class-conservation invariants where applicable.

Do not accept “new tests pass” if pre-existing focused tests in the same area now fail. The agent must run the full relevant focused test file or directory, not only newly added test functions.

If tests unrelated to the selected subtask fail because of optional local dependencies (for example CUDA, `torchaudio`, HuggingFace model availability, NLTK download, or proxy configuration), the agent must:

- distinguish environmental failures from code failures;
- report the exact failing test and root cause;
- not disable, skip, weaken, delete, or mark xfail an existing test merely to get green output;
- still run all relevant purely offline tests that can run.

============================================================
VALIDATION STANDARD
============================================================

The task is not complete merely because of:

- a new line, method, class, import, constant, or CSV filename;
- a grep result;
- successful syntax compilation;
- a successful import;
- one passing newly written test;
- a green test subset that avoids relevant existing tests.

The prompt must define observable completion criteria appropriate to the selected subtask. These should include, as applicable:

- exact generated file(s);
- exact CLI output behavior;
- actual CSV headers and representative rows;
- correct handling of empty output;
- correct classification boundaries;
- passing focused test suite;
- preservation of relevant invariants.

If local repository data are available, ask for one safe manual command that validates the changed step without launching an LLM translation backend. Use a separate temporary `--translate-dir` when appropriate so that debug output does not overwrite normal pipeline artifacts.

============================================================
ERROR HANDLING AND CODE QUALITY
============================================================

- Do not suppress failures through `analysis_options.yaml`, `pyproject.toml`, `pytest.ini`,
  `// ignore`, `// ignore_for_file`, lint suppression, broad exception swallowing,
  skipping an existing test, weakening assertions, or changing test discovery.

- Do not introduce a fallback that silently changes scientific/data semantics.
  Prefer a clear validation error or a visible warning when input schema is invalid.

- Do not add broad `except Exception: pass` blocks.

- Do not modify dependency versions, CUDA/PyTorch/HuggingFace/NLTK setup, or proxy settings unless
  the selected subtask explicitly concerns environment installation or runtime dependencies.

- Preserve backwards compatibility only when it is already an explicit repository requirement.
  Do not add a migration framework for a narrowly scoped schema change.

============================================================
MANDATORY PROMPT STRUCTURE
============================================================

Produce the CLI prompt with these sections, adapted to the selected subtask:

1. `TOOL-CALL DISCIPLINE`
   Include the exact mandatory block supplied below.

2. `TASK`
   State the selected subtask in one or two sentences.

3. `SCOPE`
   List in-scope full repository-relative files and clearly list important out-of-scope areas.

4. `CURRENT PROBLEM`
   Explain the concrete bug or missing behavior, including why the current code produces it.

5. `REQUIRED IMPLEMENTATION`
   Give ordered, small implementation requirements.
   Include schemas, invariants, and code fragments where they are critical.

6. `TESTS`
   Specify regression tests and exact behavioral assertions.

7. `OBSERVABLE COMPLETION CRITERIA`
   State what a human can verify from test output, console output, and generated CSVs.

8. `VALIDATION COMMANDS`
   Give exact commands, preferably focused pytest commands and a safe relevant CLI command.

9. `FINAL REPORT`
   Require exactly these subsections:
   - `Changed files`
   - `Implemented requirements`
   - `Tests run`
   - `Known limitations`

============================================================
TOOL-CALL DISCIPLINE
============================================================

- Use the editor's native structured tool interface; do not print pseudo-tool calls,
  XML tool calls, Markdown JSON blocks, or narration instead of invoking a tool.
- For each edit, make one small atomic replacement in one full-path file.
- Before editing, read the exact target fragment from the file.
- The edit must send all required fields in the tool schema:
  filePath, oldString, newString.
- Use camelCase schema names exactly; do not use file_path, old_string, or new_string.
- If a tool validation error occurs, retry once with the exact required schema.
  Do not repeat narration or issue another empty tool call.
- Prefer several small edits over one large write containing a whole long source file.
