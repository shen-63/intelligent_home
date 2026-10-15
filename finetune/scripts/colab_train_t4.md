# =================================================================
# Colab T4 免费卡一键训练（推荐用这个，你本机 3050 4G 容易爆）
# 在 colab 新建一个 notebook，全部粘贴到 cell 里执行就行。
# 训练完后会把 saves/qwen2.5-1.5b-nlu-lora 自动打包成 lora_adapter.zip，
# 下到本机后用 llamafactory-cli export 合并即可。
# =================================================================
#
# !nvidia-smi
# %cd /content
# !git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
# %cd LLaMA-Factory
# !pip install -e .[torch,metrics] -q
# !pip install accelerate transformers datasets peft bitsandbytes trl -q
#
# # ---- 把你项目里的 train 配置 + 数据集 上传到 Colab ----
# # （推荐：把整个 intelligent_home.zip 上传到 Google Drive，然后 mount）
# from google.colab import drive
# drive.mount('/content/drive')
# !unzip -q /content/drive/MyDrive/intelligent_home.zip -d /content/proj
# %cd /content/proj
#
# # ---- 开始 LoRA SFT ----
# !HF_TOKEN="" llamafactory-cli train finetune/train/configs/lora_sft_qwen1.5b.yaml
#
# # ---- 导出 LoRA adapter 下到本机 ----
# !cd /content/proj && zip -r /content/lora_adapter.zip finetune/train/saves/qwen2.5-1.5b-nlu-lora
# from google.colab import files
# files.download('/content/lora_adapter.zip')
