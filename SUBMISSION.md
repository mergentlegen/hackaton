# Сдача: hack-1d15670c-team

## Ссылки
- Репозиторий, финальный коммит: https://github.com/BAITC-Hacks/hack-1d15670c-team/commit/4141e7dfe0a9c9a6c9443428f6b66649070b85a0
- Публичный ноутбук Kaggle, сохранённая версия (`COMMIT` в нём = коммит выше; та же ссылка есть в
  README): https://www.kaggle.com/code/nurale/soup-kernel-t4-bench
- Демо-видео (до 3 минут): <ссылка на видео>
- Презентация: <ссылка на презентацию>

## Команда
| Имя | Роль | GitHub | Kaggle | Контакт (капитан) |
|---|---|---|---|---|
| Нурсаят | Капитан, ядра Triton и замеры | nursayatdev | nurale | @nursayat |
| Мерген Тлеген | Ядра Triton, память и скорость обучения | mergentlegen | mergentlegen | |

## Наши ядра
| Ядро (`@triton.jit`) | Файл | Что заменяет | Доля времени обучения (проверка 4) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/nf4.py` | Декодирование NF4-весов MLP в bitsandbytes (forward и backward) | 6,8% |
| `rmsnorm_kernel` | `solution/rmsnorm.py` | RMSNorm и сложение с residual (forward и backward) | 2,3% |
| `qkv_kernel` | `solution/attention.py` | RoPE, приведение к fp16, раскладку q/k/v по головам и GQA (forward и backward) | 1,6% |
| `ce_kernel` | `solution/kernels.py` | Кросс-энтропию, энтропию и argmax по словарю, градиент по логитам | 1,1% |
| `silu_mul_kernel` | `solution/kernels.py` | SwiGLU в MLP — повторяет работу примера, своим не считаем | 5,5% |
| **Всего** | | | **17,3%** (без `silu_mul_kernel` — **11,9%**, порог 5%) |

Идеи из других работ, со ссылками (код написали сами):
- линейный слой + кросс-энтропия кусками: [Liger-Kernel](https://github.com/linkedin/Liger-Kernel), [Cut Your Losses](https://github.com/apple/ml-cross-entropy);
- выгрузка активаций: [Unsloth](https://github.com/unslothai/unsloth), [torchtune](https://github.com/pytorch/torchtune);
- RMSNorm на Triton: [туториал Triton](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html);
- varlen attention с `cu_seqlens`: [FlashAttention](https://github.com/Dao-AILab/flash-attention) (вызываем memory-efficient attention из PyTorch);
- формат NF4: [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) / QLoRA.

## Результаты на Kaggle T4 (из `runs/result.json` и `runs/checks.json`, `--repeats 2`)
| | База Soup | Наше решение | Отношение |
|---|---|---|---|
| Время `soup train`, с | 87,06 | 55,89 | ускорение 1,558x |
| Пиковая память GPU, МиБ | 14345 | 1809 | база/наше 7,930x |
| Eval loss (публичные строки) | 2,04968 | 2,04982 | +0,007% к базе |

- [x] Все 10 проверок `harness/check_rules.py` пройдены (10/10)
- [x] README.md переписан под наше решение: что сделали, как запустить, результаты, ограничения

## Что покрывает презентация
Какие операции ускорили и почему, как работают ядра, результаты против базы (скорость, память,
качество), ограничения и следующие шаги.

## Ограничения
Подробно — в разделе 4 README. Коротко: около 21 с команды уходят вне цикла обучения одинаково у
базы и у решения, шаг упирается в fp16 GEMM cuBLAS, по памяти мы близко к полу (CUDA-контекст и
4-битные веса, ~1,4 ГБ), всё настроено под конфигурацию задачи на T4, а другие случаи уходят в
исходный код.
