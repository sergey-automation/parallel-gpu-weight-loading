# Parallel GPU Weight Loading

**Parallel GPU Weight Loading for
[llama.cpp](https://github.com/ggml-org/llama.cpp) and
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)**

An independently developed, optional multi-GPU model-weight loading
modification. It is **not a built-in feature of the upstream projects**.
In the tested multi-GPU configurations, enabling `--parallel-load`
reduced model loading time by **2.88--3.71×** without changing prefill
or decode throughput after the model was loaded.

## Implementations

The implementation source is maintained in the existing [TurboPrefill
repository](https://github.com/sergey-automation/TurboPrefill); no
duplicate copies of the source trees are stored here.

-   **llama.cpp b11429:**
    [TurboPrefill_llama.cpp_b11429_v2.2m](https://github.com/sergey-automation/TurboPrefill/tree/main/TurboPrefill_llama.cpp_b11429_v2.2m)
-   **ik_llama.cpp build 5005:**
    [TurboPrefill_ik_llama.cpp_5005_V6.9.1](https://github.com/sergey-automation/TurboPrefill/tree/main/TurboPrefill_ik_llama.cpp_5005_V6.9.1)

The measurements below were made with the **llama.cpp b11429 v2.2m**
implementation. The ik_llama.cpp link provides the corresponding port;
the numeric results below should not be interpreted as an independent
ik_llama.cpp benchmark.

## Test system and methodology

  ---------------------------------------------------------------------------
  Component                           Tested configuration
  ----------------------------------- ---------------------------------------
  GPUs                                NVIDIA P104-100 8 GB; 5, 6 or 12 GPUs
                                      used per model depending on
                                      configuration

  GPU power limit                     **90 W per GPU**, versus the **stock
                                      180 W** power limit

  GPU connection                      **PCIe Gen1 ×1 per GPU** through risers

  Model storage                       **Samsung PM981 512 GB NVMe SSD**,
                                      mounted at `/mnt/pm981`; GGUF files
                                      under `/mnt/pm981/models`

  Software                            Modified llama.cpp b11429,
                                      `TurboPrefill_llama.cpp_b11429_v2.2m`

  Multi-GPU mode                      `-sm layer`

  Date                                2026-10-15

  Prompt lengths                      512, 1024, 2048 and 8192 tokens

  Generation                          128 output tokens requested per test
  ---------------------------------------------------------------------------

The hardware deliberately represents a bandwidth-constrained multi-GPU
setup. **PCIe Gen1 ×1 is a major bottleneck for model-weight
transfers**, and the P104-100 cards were additionally limited to **90 W
instead of their stock 180 W power limit**. The results therefore
describe this exact test system; they are not a universal loading-speed
or inference-speed guarantee.

Each benchmark directory contains the raw benchmark output,
`results.csv`, `summary.md`, exact launch settings (`launch.json`),
server logs, request/response data and per-context measurements. Rates
are in tokens per second (`tok/s`).

## 1. Parallel GPU Weight Loading --- primary result

This is the main experiment.

**Controlled comparison:** the targeted change is only
`--parallel-load`. Both runs have **TurboPrefill OFF** and **MTP OFF**.

-   Parallel GPU Weight Loading **ON**:
    [`parallel_on_turboprefill_off_mtp_off/`](benchmark-results/parallel_on_turboprefill_off_mtp_off/)
-   Parallel GPU Weight Loading **OFF**:
    [`parallel_off_turboprefill_off_mtp_off/`](benchmark-results/parallel_off_turboprefill_off_mtp_off/)

### Full model loading times

The times below measure startup until the server is ready, including
model loading and initialization. They are not a measurement of SSD
throughput alone.

  --------------------------------------------------------------------------------------
  Model                        GPUs  Parallel ON Parallel OFF       **Time   **Speedup**
                                                                   saved** 
  -------------------- ------------ ------------ ------------ ------------ -------------
  Qwen3.8-27B Q8_0                6  **48.71 s**     162.22 s   **113.51 s     **3.33×**
                                          (0:49)       (2:42)     (1:54)** 

  Qwen3.8-27B Q4_K_M              5  **34.09 s**      98.14 s    **64.05 s     **2.88×**
                                          (0:34)       (1:38)     (1:04)** 

  Qwen3.8-Flash-Next             12 **138.49 s**     513.53 s   **375.04 s     **3.71×**
  Q4_K_XL                                 (2:18)       (8:34)     (6:15)** 

  Qwen3.6-35B-A3B Q8_0            6  **72.59 s**     212.25 s   **139.66 s     **2.92×**
                                          (1:13)       (3:32)     (2:20)** 

  **Total, four                 ---   **293.88 s   **986.15 s   **692.27 s     **3.36×**
  sequential loads**                    (4:54)**    (16:26)**    (11:32)** 
  --------------------------------------------------------------------------------------

### Visual comparison

Each `█` represents approximately 15 seconds. Shorter is better.

``` text
Q8     OFF  ███████████                         2:42
       ON   ███                                 0:49

Q4     OFF  ███████                             1:38
       ON   ██                                  0:34

Flash  OFF  ██████████████████████████████████  8:34
       ON   █████████                           2:18

35B    OFF  ██████████████                      3:32
       ON   █████                               1:13
```

**Total: 16m 26s → 4m 54s. Time saved: 11m 32s.**

The largest absolute improvement was Qwen3.8-Flash-Next: **6m 15s saved
per model load**.

### Parallel loading does not accelerate inference

Parallel GPU Weight Loading acts during **startup/model loading only**.
Once the model is loaded, prefill and decode throughput remain within
normal measurement variation.

At 8192 prompt tokens:

  Model                          Prefill ON   Prefill OFF   Decode ON   Decode OFF
  ---------------------------- ------------ ------------- ----------- ------------
  Qwen3.8-27B Q8_0                    96.54         96.47        7.01         7.01
  Qwen3.8-27B Q4_K_M                  89.66         89.67        8.69         8.69
  Qwen3.8-Flash-Next Q4_K_XL          33.91         33.96       10.63        10.65
  Qwen3.6-35B-A3B Q8_0               197.54        198.20       29.21        28.46

Across the complete 512/1024/2048/8192 comparison, the average absolute
difference was approximately **0.24% for prefill** and **0.57% for
decode**.

**Result:** on this PCIe Gen1 ×1 system, Parallel GPU Weight Loading
reduced startup time by approximately **2.9--3.7×** while leaving
inference performance effectively unchanged.

## 2. TurboPrefill --- prefill acceleration

TurboPrefill is a separate mechanism from Parallel GPU Weight Loading.

For the clean TurboPrefill comparison, Parallel GPU Weight Loading stays
**ON** and MTP stays **OFF**:

-   TurboPrefill **ON**:
    [`parallel_on_turboprefill_on_mtp_off/`](benchmark-results/parallel_on_turboprefill_on_mtp_off/)
-   TurboPrefill **OFF**:
    [`parallel_on_turboprefill_off_mtp_off/`](benchmark-results/parallel_on_turboprefill_off_mtp_off/)

TurboPrefill was active for **Qwen3.8-27B Q8_0**, **Qwen3.8-27B Q4_K_M**
and **Qwen3.6-35B-A3B Q8_0**. Qwen3.8-Flash-Next was run without
TurboPrefill and is not included in the TurboPrefill speedup table.

### Prefill speedup

  Model                          512        1024        2048        8192
  ---------------------- ----------- ----------- ----------- -----------
  Qwen3.8-27B Q8_0         **1.60×**   **2.08×**   **2.59×**   **3.28×**
  Qwen3.8-27B Q4_K_M       **1.59×**   **2.01×**   **2.47×**   **3.03×**
  Qwen3.6-35B-A3B Q8_0     **1.49×**   **2.15×**   **2.49×**   **3.06×**

### Actual prefill throughput

  Model   TurboPrefill        512     1024     2048     8192
  ------- -------------- -------- -------- -------- --------
  Q8      ON               122.39   188.01   248.57   317.05
  Q8      OFF               76.28    90.56    96.11    96.54
  Q4      ON               114.54   171.03   222.34   271.51
  Q4      OFF               71.93    85.00    89.88    89.66
  35B     ON               230.80   423.42   510.40   604.67
  35B     OFF              154.86   196.59   204.91   197.54

### Decode is unaffected by TurboPrefill

With MTP disabled in both runs, decode throughput stayed within
measurement variation. At 8192 tokens:

  Model                    Decode, TurboPrefill ON   Decode, TurboPrefill OFF
  ---------------------- ------------------------- --------------------------
  Qwen3.8-27B Q8_0                            7.01                       7.01
  Qwen3.8-27B Q4_K_M                          8.68                       8.69
  Qwen3.6-35B-A3B Q8_0                       28.82                      29.21

**Result:** TurboPrefill accelerates **prefill**, increasingly strongly
at longer prompts, while decode is unchanged within measurement
variability.

## 3. MTP --- decode acceleration with measurable prefill overhead

The new benchmark completes the full 2×2 TurboPrefill/MTP matrix while
keeping Parallel GPU Weight Loading **ON** in every run:

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                          **MTP OFF**                                                                                          **MTP ON**
  ----------------------- ---------------------------------------------------------------------------------------------------- --------------------------------------------------------------------------------------------------
  **TurboPrefill OFF**    [`parallel_on_turboprefill_off_mtp_off/`](benchmark-results/parallel_on_turboprefill_off_mtp_off/)   [`parallel_on_turboprefill_off_mtp_on/`](benchmark-results/parallel_on_turboprefill_off_mtp_on/)

  **TurboPrefill ON**     [`parallel_on_turboprefill_on_mtp_off/`](benchmark-results/parallel_on_turboprefill_on_mtp_off/)     [`parallel_on_turboprefill_on_mtp_on/`](benchmark-results/parallel_on_turboprefill_on_mtp_on/)
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

MTP was enabled for **Qwen3.8-27B Q8_0** and **Qwen3.8-27B Q4_K_M**
with:

``` text
--spec-type draft-mtp --spec-draft-n-max 3
```

Flash and Qwen3.6-35B-A3B remained MTP OFF.

### MTP without TurboPrefill

This comparison isolates MTP in standard prefill mode:

-   MTP OFF:
    [`parallel_on_turboprefill_off_mtp_off/`](benchmark-results/parallel_on_turboprefill_off_mtp_off/)
-   MTP ON:
    [`parallel_on_turboprefill_off_mtp_on/`](benchmark-results/parallel_on_turboprefill_off_mtp_on/)

#### Decode speedup

  Model     Context   MTP OFF   MTP ON   **Speedup**
  ------- --------- --------- -------- -------------
  Q8            512      7.21    14.28     **1.98×**
  Q8           1024      7.15    10.55     **1.47×**
  Q8           2048      7.09    14.93     **2.11×**
  Q8           8192      7.01    14.40     **2.06×**
  Q4            512      8.97    17.14     **1.91×**
  Q4           1024      9.02    16.53     **1.83×**
  Q4           2048      8.83    16.22     **1.84×**
  Q4           8192      8.69    15.24     **1.75×**

#### Prefill overhead

  Model     Context   MTP OFF   MTP ON       Change
  ------- --------- --------- -------- ------------
  Q8            512     76.28    67.69   **−11.3%**
  Q8           1024     90.56    85.08    **−6.1%**
  Q8           2048     96.11    90.38    **−6.0%**
  Q8           8192     96.54    90.14    **−6.6%**
  Q4            512     71.93    65.93    **−8.3%**
  Q4           1024     85.00    80.05    **−5.8%**
  Q4           2048     89.88    84.59    **−5.9%**
  Q4           8192     89.66    84.12    **−6.2%**

So in standard mode, MTP improved decode by **1.47--2.11×** in these
tests while reducing prefill throughput by roughly **6--11%**.

### MTP with TurboPrefill

This comparison isolates MTP while TurboPrefill is active:

-   MTP OFF:
    [`parallel_on_turboprefill_on_mtp_off/`](benchmark-results/parallel_on_turboprefill_on_mtp_off/)
-   MTP ON:
    [`parallel_on_turboprefill_on_mtp_on/`](benchmark-results/parallel_on_turboprefill_on_mtp_on/)

#### Decode speedup

  Model     Context   MTP OFF   MTP ON   **Speedup**
  ------- --------- --------- -------- -------------
  Q8            512      7.21    16.45     **2.28×**
  Q8           1024      7.14    12.25     **1.72×**
  Q8           2048      7.14    15.52     **2.17×**
  Q8           8192      7.01    13.39     **1.91×**
  Q4            512      8.97    17.35     **1.94×**
  Q4           1024      9.04    17.33     **1.92×**
  Q4           2048      8.98    16.84     **1.88×**
  Q4           8192      8.68    15.92     **1.83×**

#### Prefill overhead

  Model     Context   MTP OFF   MTP ON       Change
  ------- --------- --------- -------- ------------
  Q8            512    122.39   106.78   **−12.8%**
  Q8           1024    188.01   166.13   **−11.6%**
  Q8           2048    248.57   214.75   **−13.6%**
  Q8           8192    317.05   264.63   **−16.5%**
  Q4            512    114.54   101.79   **−11.1%**
  Q4           1024    171.03   153.07   **−10.5%**
  Q4           2048    222.34   195.06   **−12.3%**
  Q4           8192    271.51   232.03   **−14.5%**

With TurboPrefill active, MTP improved decode by **1.72--2.28×** for Q8
and **1.83--1.94×** for Q4, with a prefill cost of approximately
**10--17%**.

### TurboPrefill + MTP still beats standard prefill

The MTP overhead does not erase the TurboPrefill gain. Comparing
**TurboPrefill ON + MTP ON** against **TurboPrefill OFF + MTP OFF**:

  -------------------------------------------------------------------------
  Model              Context       Standard   TurboPrefill +    **Speedup**
                                    prefill              MTP 
  ----------- -------------- -------------- ---------------- --------------
  Q8                     512          76.28           106.78      **1.40×**

  Q8                    1024          90.56           166.13      **1.83×**

  Q8                    2048          96.11           214.75      **2.23×**

  Q8                    8192          96.54           264.63      **2.74×**

  Q4                     512          71.93           101.79      **1.42×**

  Q4                    1024          85.00           153.07      **1.80×**

  Q4                    2048          89.88           195.06      **2.17×**

  Q4                    8192          89.66           232.03      **2.59×**
  -------------------------------------------------------------------------

**Result:** MTP consistently trades some prefill throughput for much
faster decode. That behavior is visible both with and without
TurboPrefill. TurboPrefill + MTP remains substantially faster at prefill
than the standard TurboPrefill-OFF/MTP-OFF configuration at every tested
prompt length.

## 4. Recommended launch commands

These are the recommended tested configurations. Parallel GPU Weight
Loading is enabled for every model. TurboPrefill and MTP are enabled
only where they were used successfully in the working setup.

### Qwen3.8-27B-Q8_0 --- port 8083

``` bash
cd /home/serg/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m && \
  env -u LD_PRELOAD TURBOPREFILL=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 LD_LIBRARY_PATH=/home/serg/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m/build/bin \
  /home/serg/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m/build/bin/llama-server \
    -m /mnt/pm981/models/Qwen3.8-27B-Q8_0.gguf \
    --mmproj /mnt/pm981/models/Qwen3.8-27B-mmproj-BF16.gguf \
    --alias Qwen3.8-27B-Q8 \
    --host 0.0.0.0 --port 8083 -ngl 999 \
    -c 262083 -b 4096 -ub 32 -np 1 -ctk f16 -ctv f16 \
    -sm layer -ts 9/12/11/11/12/9 \
    --parallel-load \
    --spec-type draft-mtp --spec-draft-n-max 3 \
    -lv 4 --reasoning-format deepseek -rea on --metrics --no-warmup
```

### Qwen3.8-27B-Q4_K_M --- port 8084

``` bash
cd /home/serg/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m && \
  env -u LD_PRELOAD TURBOPREFILL=1 CUDA_VISIBLE_DEVICES=7,8,9,10,11 LD_LIBRARY_PATH=/home/serg/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m/build/bin \
  /home/serg/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m/build/bin/llama-server \
    -m /mnt/pm981/models/Qwen3.8-27B-Q4_K_M.gguf \
    --mmproj /mnt/pm981/models/Qwen3.8-27B-mmproj-BF16.gguf \
    --alias Qwen3.8-27B-Q4 \
    --host 0.0.0.0 --port 8084 -ngl 999 \
    -c 262084 -b 4096 -ub 32 -np 1 -ctk f16 -ctv f16 \
    -sm layer -ts 10/12/12/12/10 \
    --parallel-load \
    --spec-type draft-mtp --spec-draft-n-max 3 \
    -lv 4 --reasoning-format deepseek -rea on --metrics --no-warmup
```

### Qwen3.8-Flash-Next-UD-Q4_K_XL --- port 8085

``` bash
cd /home/serg/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m && \
  env -u LD_PRELOAD CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 LD_LIBRARY_PATH=/home/serg/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m/build/bin \
  /home/serg/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m/build/bin/llama-server \
    -m /mnt/pm981/models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    --mmproj /mnt/pm981/models/UD-Q4_K_XL/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
    --alias Qwen3.8-Flash-Next-UD-Q4_K_XL \
    --host 0.0.0.0 --port 8085 \
    -c 262085 -b 4096 -ub 32 -np 1 -ctk f16 -ctv f16 \
    -sm layer \
    -fit on -fitt 128 -lm mmap -lzm on \
    --parallel-load \
    -lv 4 --reasoning-format deepseek -rea on --metrics --no-warmup
```

### Qwen3.6-35B-A3B-abliterated-v4-Q8_0 --- port 8086

``` bash
cd /home/serg/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m && \
  env -u LD_PRELOAD TURBOPREFILL=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 LD_LIBRARY_PATH=/home/serg/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m/build/bin \
  /home/serg/workspace/versions/TurboPrefill_llama.cpp_b11429_v2.2m/build/bin/llama-server \
    -m /mnt/pm981/models/Qwen3.6-35B-A3B-abliterated-v4-Q8_0.gguf \
    --host 0.0.0.0 --port 8086 \
    -lv 4 -ngl 999 -c 262086 -b 4096 -ub 32 -np 1 \
    -ctk f16 -ctv f16 -sm layer \
    --parallel-load \
    -ts 4/4/8/8/8/8 \
    --no-warmup \
    --mmproj /mnt/pm981/models/Qwen3.6-35B-A3B-mmproj-BF16.gguf \
    --spec-type none --metrics
```

## 5. Published benchmark results

The repository contains the five successful configurations needed to
isolate Parallel GPU Weight Loading and complete the TurboPrefill/MTP
matrix:

  ------------------------------------------------------------------------------------------------------------------------------------------------------------
  Directory                                                                                              Parallel GPU      TurboPrefill      MTP
                                                                                                         Weight Loading                      
  ------------------------------------------------------------------------------------------------------ ----------------- ----------------- -----------------
  [`parallel_on_turboprefill_on_mtp_on/`](benchmark-results/parallel_on_turboprefill_on_mtp_on/)         ON                ON where used     ON for Q8/Q4

  [`parallel_on_turboprefill_on_mtp_off/`](benchmark-results/parallel_on_turboprefill_on_mtp_off/)       ON                ON where used     OFF

  [`parallel_on_turboprefill_off_mtp_on/`](benchmark-results/parallel_on_turboprefill_off_mtp_on/)       ON                OFF               ON for Q8/Q4

  [`parallel_on_turboprefill_off_mtp_off/`](benchmark-results/parallel_on_turboprefill_off_mtp_off/)     ON                OFF               OFF

  [`parallel_off_turboprefill_off_mtp_off/`](benchmark-results/parallel_off_turboprefill_off_mtp_off/)   OFF               OFF               OFF
  ------------------------------------------------------------------------------------------------------------------------------------------------------------

Repository layout:

``` text
parallel-gpu-weight-loading/
├── README.md
├── LICENSE
└── benchmark-results/
    ├── parallel_on_turboprefill_on_mtp_on/
    ├── parallel_on_turboprefill_on_mtp_off/
    ├── parallel_on_turboprefill_off_mtp_on/
    ├── parallel_on_turboprefill_off_mtp_off/
    └── parallel_off_turboprefill_off_mtp_off/
```

No benchmark ZIP archives are required in the repository: the result
directories are published directly so `summary.md`, `results.csv`,
`launch.json` and the raw logs can be inspected in GitHub.

## Summary

-   **Parallel GPU Weight Loading** reduced model startup/loading time
    by **2.88--3.71×** on the tested PCIe Gen1 ×1 multi-GPU system,
    saving **11m 32s** across four sequential model loads. It does
    **not** accelerate prefill or decode.
-   **TurboPrefill** accelerated prefill by up to **3.28×** at 8192
    tokens in the tested configurations and did not materially change
    decode speed.
-   **MTP** accelerated decode while imposing a measurable prefill
    penalty both with and without TurboPrefill.
-   **TurboPrefill + MTP** still produced substantially faster prefill
    than standard TurboPrefill-OFF/MTP-OFF operation at every tested
    context length.
-   The GPUs were operated at a **90 W power limit instead of the stock
    180 W limit**, and connected through **PCIe Gen1 ×1**, so benchmark
    results should be interpreted in the context of this deliberately
    constrained test platform.
