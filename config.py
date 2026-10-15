import os

# 从项目根目录的 .env 加载环境变量（该文件已被 .gitignore 排除，不会提交到仓库）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

SILICONFLOW_MODEL = os.environ.get("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct")

RULE_CONFIDENCE_THRESHOLD = 0.7
LLM_TIMEOUT = 30
LLM_MAX_RETRIES = 1

# ==================== MQTT Broker 配置 ====================
MQTT_BROKER_HOST = os.environ.get("MQTT_BROKER_HOST", "127.0.0.1")
MQTT_BROKER_PORT = int(os.environ.get("MQTT_BROKER_PORT", "1883"))
MQTT_USERNAME = os.environ.get("MQTT_USERNAME", "")
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD", "")
MQTT_ENABLED = os.environ.get("MQTT_ENABLED", "true").lower() == "true"
MQTT_TOPIC_PREFIX = os.environ.get("MQTT_TOPIC_PREFIX", "home")