@echo off
REM =================================================================
REM 阶段 B/C/D 一键脚手架（Windows PowerShell 下双击或 cmd 跑）
REM 用法：
REM   scripts\01_train_lora_colab_or_gpu.cmd
REM 注意：本脚手架的命令，是在你"已经装好 llamafactory-cli"的环境里执行
REM      （Colab 里用 /content 目录，或本机 Python env）
REM 这里只打印命令，帮你省时间。
REM =================================================================

echo.
echo [01] LoRA SFT 训练（Qwen2.5-1.5B-Instruct, QLoRA 4bit, 3 epoch）
echo.
echo   llamafactory-cli train finetune/train/configs/lora_sft_qwen1.5b.yaml
echo.
echo ================================================================
echo [02] 训练完后，把 LoRA adapter 合并回基座（导出 HF 格式）
echo.
echo   llamafactory-cli export finetune/train/configs/export_merge_lora.yaml
echo.
echo ================================================================
echo [03] 量化：HF -> F16 GGUF -> Q4_K_M
echo   （需要 llama.cpp 编译好的 convert_hf_to_gguf.py / llama-quantize.exe）
echo.
echo   python ../../llama.cpp/convert_hf_to_gguf.py ^
echo       finetune/train/saves/qwen2.5-1.5b-nlu-merged ^
echo       --outtype f16 --outfile C:/models/qwen2.5-1.5b-nlu-f16.gguf
echo.
echo   C:/tools/llama-b9878-bin-win-cpu-x64/bin/llama-quantize.exe ^
echo       C:/models/qwen2.5-1.5b-nlu-f16.gguf ^
echo       C:/models/qwen2.5-1.5b-nlu-q4_k_m.gguf q4_k_m
echo.
echo ================================================================
echo [04] Ollama create（本地部署）
echo.
echo   cd c:\Users\shy\Desktop\intelligent_home
echo   (Get-Content finetune\deploy\ollama\Modelfile.template -Raw) ^
echo       -replace '{{GGUF_PATH}}','C:/models/qwen2.5-1.5b-nlu-q4_k_m.gguf' ^| ^
echo       Set-Content finetune\deploy\ollama\Modelfile -Encoding UTF8
echo   ollama create qwen2.5-1.5b-nlu:q4_k_m -f finetune\deploy\ollama\Modelfile
echo   ollama run qwen2.5-1.5b-nlu:q4_k_m
echo.
echo ================================================================
echo [05] 阶段 C：跑 4 组基线评估
echo.
echo   set SILICONFLOW_API_KEY=你的key
echo   python -m finetune.eval.eval_nlu --backends rule,sfapi,base,sft --limit 150
echo.
echo ================================================================
echo [06] 自测 Ollama NLU Wrapper（确保能被 smart_home_nlu.py 无缝替换）
echo.
echo   python -m finetune.deploy.llm_nlu_local_ollama
echo.
pause
