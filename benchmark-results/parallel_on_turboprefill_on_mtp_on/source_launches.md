# OpenCode — актуальные локальные модели (b11429_v2.2m)

Рабочая сборка:

``` bash
~/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m
```

Все команды ниже перед запуском очищают **только те GPU, которые нужны
конкретной модели**. Остановка сервера выполняется через `SIGINT`.

## Сводка конфигураций

| Модель                              | Порт | Контекст `-c` | GPU                       | Параллельная загрузка |
|:------------------------------------|-----:|--------------:|:--------------------------|:----------------------|
| Qwen3.8-27B-Q8_0                    | 8083 |        262083 | 0–5 (6 × P104-100 8 ГБ)   | `--parallel-load`     |
| Qwen3.8-27B-Q4_K_M                  | 8084 |        262084 | 7–11 (5 × P104-100 8 ГБ)  | `--parallel-load`     |
| Qwen3.8-Flash-Next-UD-Q4_K_XL       | 8085 |        262085 | 0–11 (12 × P104-100 8 ГБ) | `--parallel-load`     |
| Qwen3.6-35B-A3B-abliterated-v4-Q8_0 | 8086 |        262086 | 0–5 (6 × P104-100 8 ГБ)   | `--parallel-load`     |

`--override-kv llama.context_length=...` удалён. Все указанные значения
`-c` ниже предполагаемого нативного предела 262144 токена. Проверить
реальный `n_ctx_train` можно по логу загрузки. Внутренний `n_ctx` может
округляться сборкой llama.cpp.

**Важно:** перед каждым запуском скрипт отправляет `SIGKILL` процессам,
использующим выбранные GPU. Если один сервер занимает часть этих GPU, он
тоже будет остановлен, даже если использует и другие карты.

------------------------------------------------------------------------

## Изменения для следующего теста

- Для всех моделей: `-b 4096`, `-ctk f16 -ctv f16`, `--parallel-load`,
  `--metrics`, `--no-warmup`.
- У Qwen3.6-35B-A3B удалён явный `--flash-attn auto`: по умолчанию
  используется `auto`.
- У Flash-Next удалены **оба** явных аргумента `-fit on -fitt 128`.
  **Важно:** это не выключает fit. По умолчанию `-fit on`, а `-fitt`
  становится 1024 МиБ вместо 128 МиБ. Время загрузки и успешность
  запуска надо проверить. Чтобы отдельно проверить запуск *без механизма
  fit*, используется `-fit off`; это другой эксперимент.
- У Flash-Next увеличение `-b` с 512 до 4096 также меняет условия теста.
  Для изолированного сравнения влияния fit лучше сохранить одинаковые
  `-b` и `--no-warmup` в обоих замерах.
- У Flash-Next по-прежнему нет явно заданных `-ngl` и `-ts`: размещение
  слоёв зависит от автоматической подгонки.

## Общая проверка GPU

``` bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
```

------------------------------------------------------------------------

------------------------------------------------------------------------

## Qwen3.8-27B-Q8_0 — порт 8083 — GPU 0–5

### Запуск

``` bash
cd ~/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m && PIDS="$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | while IFS=',' read -r pid uuid; do pid="$(echo "$pid" | xargs)"; uuid="$(echo "$uuid" | xargs)"; for g in 0 1 2 3 4 5; do [ "$uuid" = "$(nvidia-smi -i "$g" --query-gpu=uuid --format=csv,noheader)" ] && echo "$pid"; done; done | sort -u)" && [ -z "$PIDS" ] || kill -KILL $PIDS; sleep 2; LOG="qwen38_q8_6gpu_parallel_$(date +%Y%m%d_%H%M%S).log"; echo "START $(date '+%F %T.%N') EPOCH=$(date +%s.%N)" > "$LOG"; nohup env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 TURBOPREFILL=1 LD_LIBRARY_PATH="$PWD/build/bin" "$PWD/build/bin/llama-server" -m /mnt/pm981/models/Qwen3.8-27B-Q8_0.gguf --mmproj /mnt/pm981/models/Qwen3.8-27B-mmproj-BF16.gguf --alias Qwen3.8-27B-Q8 --host 0.0.0.0 --port 8083 -ngl 999 -c 262083 -b 4096 -ub 32 -np 1 -ctk f16 -ctv f16 -sm layer -ts "9/12/11/11/12/9" --parallel-load --spec-type draft-mtp --spec-draft-n-max 3 -lv 4 --reasoning-format deepseek -rea on --metrics --no-warmup >> "$LOG" 2>&1 & echo "PID=$! LOG=$LOG"
```

### Лог

``` bash
tail -n 200 -f "$(ls -1t qwen38_q8_6gpu_parallel_*.log | head -1)"
```

### Остановка

``` bash
pkill -INT -f '[l]lama-server.*--port 8083'
```

------------------------------------------------------------------------

## Qwen3.8-27B-Q4_K_M — порт 8084 — GPU 7–11

### Запуск

``` bash
cd ~/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m && PIDS="$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | while IFS=',' read -r pid uuid; do pid="$(echo "$pid" | xargs)"; uuid="$(echo "$uuid" | xargs)"; for g in 7 8 9 10 11; do [ "$uuid" = "$(nvidia-smi -i "$g" --query-gpu=uuid --format=csv,noheader)" ] && echo "$pid"; done; done | sort -u)" && [ -z "$PIDS" ] || kill -KILL $PIDS; sleep 2; LOG="qwen38_q4_5gpu_parallel_$(date +%Y%m%d_%H%M%S).log"; echo "START $(date '+%F %T.%N') EPOCH=$(date +%s.%N)" > "$LOG"; nohup env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=7,8,9,10,11 TURBOPREFILL=1 LD_LIBRARY_PATH="$PWD/build/bin" "$PWD/build/bin/llama-server" -m /mnt/pm981/models/Qwen3.8-27B-Q4_K_M.gguf --mmproj /mnt/pm981/models/Qwen3.8-27B-mmproj-BF16.gguf --alias Qwen3.8-27B-Q4 --host 0.0.0.0 --port 8084 -ngl 999 -c 262084 -b 4096 -ub 32 -np 1 -ctk f16 -ctv f16 -sm layer -ts "10/12/12/12/10" --parallel-load --spec-type draft-mtp --spec-draft-n-max 3 -lv 4 --reasoning-format deepseek -rea on --metrics --no-warmup >> "$LOG" 2>&1 & echo "PID=$! LOG=$LOG"
```

### Лог

``` bash
tail -n 200 -f "$(ls -1t qwen38_q4_5gpu_parallel_*.log | head -1)"
```

### Остановка

``` bash
pkill -INT -f '[l]lama-server.*--port 8084'
```

------------------------------------------------------------------------

## Qwen3.8-Flash-Next-UD-Q4_K_XL — порт 8085 — GPU 0–11

Рабочий вариант: **без MTP и без TurboPrefill**, 12 GPU.

Из прежних замеров: загрузка 4:04, prefill ≈36 tok/s, decode ≈10 tok/s;
также был запуск 2:22, prefill ≈34 tok/s, decode ≈10 tok/s при 12/12.

### Запуск

``` bash
cd ~/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m && PIDS="$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | while IFS=',' read -r pid uuid; do pid="$(echo "$pid" | xargs)"; uuid="$(echo "$uuid" | xargs)"; for g in 0 1 2 3 4 5 6 7 8 9 10 11; do [ "$uuid" = "$(nvidia-smi -i "$g" --query-gpu=uuid --format=csv,noheader)" ] && echo "$pid"; done; done | sort -u)" && [ -z "$PIDS" ] || kill -KILL $PIDS; sleep 2; LOG="qwen38_flash_next_ud_q4_12gpu_nomtp_noturbo_$(date +%Y%m%d_%H%M%S).log"; echo "START $(date '+%F %T.%N') EPOCH=$(date +%s.%N)" > "$LOG"; nohup env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 LD_LIBRARY_PATH="$PWD/build/bin" "$PWD/build/bin/llama-server" -m /mnt/pm981/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --mmproj /mnt/pm981/models/UD-Q4_K_XL/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf --alias Qwen3.8-Flash-Next-UD-Q4_K_XL --host 0.0.0.0 --port 8085 -c 262085 -b 4096 -ub 32 -np 1 -ctk f16 -ctv f16 -sm layer -fit on -fitt 128 -lm mmap -lzm on --parallel-load -lv 4 --reasoning-format deepseek -rea on --metrics --no-warmup >> "$LOG" 2>&1 & echo "PID=$! LOG=$LOG"
```

### Лог

``` bash
tail -n 200 -f "$(ls -1t qwen38_flash_next_ud_q4_12gpu_nomtp_noturbo_*.log | head -1)"
```

### Остановка

``` bash
pkill -INT -f '[l]lama-server.*--port 8085'
```

------------------------------------------------------------------------

## Qwen3.6-35B-A3B-abliterated-v4-Q8_0 — порт 8086 — GPU 0–5

### Запуск

``` bash
cd ~/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m && PIDS="$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | while IFS=',' read -r pid uuid; do pid="$(echo "$pid" | xargs)"; uuid="$(echo "$uuid" | xargs)"; for g in 0 1 2 3 4 5; do [ "$uuid" = "$(nvidia-smi -i "$g" --query-gpu=uuid --format=csv,noheader)" ] && echo "$pid"; done; done | sort -u)" && [ -z "$PIDS" ] || kill -KILL $PIDS; sleep 2; LOG="qwen36_35b_a3b_q8_turbo_$(date +%Y%m%d_%H%M%S).log"; echo "START $(date '+%F %T.%N') EPOCH=$(date +%s.%N)" > "$LOG"; nohup env -u LD_PRELOAD TURBOPREFILL=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 LD_LIBRARY_PATH="$PWD/build/bin" "$PWD/build/bin/llama-server" -m /mnt/pm981/models/Qwen3.6-35B-A3B-abliterated-v4-Q8_0.gguf --host 0.0.0.0 --port 8086 -lv 4 -ngl 999 -c 262086 -b 4096 -ub 32 -np 1 -ctk f16 -ctv f16 -sm layer --parallel-load -ts 4/4/8/8/8/8 --no-warmup --mmproj /mnt/pm981/models/Qwen3.6-35B-A3B-mmproj-BF16.gguf --spec-type none --metrics >> "$LOG" 2>&1 & echo "PID=$! LOG=$LOG"
```

### Лог

``` bash
tail -n 200 -f "$(ls -1t qwen36_35b_a3b_q8_turbo_*.log | head -1)"
```

### Остановка

``` bash
pkill -INT -f '[l]lama-server.*--port 8086'
```

------------------------------------------------------------------------

## Быстрая проверка запущенных серверов

``` bash
ps -ef | grep '[l]lama-server'
```

``` bash
ss -ltnp | grep -E ':8083|:8084|:8085|:8086'
```

## Проверка памяти GPU после запуска

``` bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
```

## Примечание по 8082

`8082` не включён: `Qwen3.6-27B-Q4` сейчас отсутствует на
`/mnt/pm981/models`.
