"""
天气查询服务 - 免费天气 API 集成 + 中文化格式化 + TTS友好播报。

使用 wttr.in 免费服务，无需 API Key：
  - 全球城市支持
  - 返回 JSON 格式的实时天气数据
  - 无调用次数限制

本次重写重点：
  1. 城市名使用 extract_city() 提取到的"原始中文城市名"，不用 wttr.in 返回的拼音
     （避免出现 Changshashih / Xianggang / Beijing-shi 等尾缀/拼音污染）
  2. 天气现象翻译映射（Weather Condition Map）
  3. 风向16方位翻译映射（Wind Direction Map，兼容 8 方位）
  4. 风力 km/h → 蒲福氏 0~12 风力等级转换（中国通用）
  5. TTS 友好自然语言模板
"""

import json
import re
import math
from typing import Dict, Optional


# =============================================================================
# 一、【固定映射表】
# =============================================================================

# ---- 天气现象翻译（wttr.in 的 weatherDesc 返回值 -> 标准中文 2~4 字描述） ----
# 参考 wttr.in 源码的 weatherDesc 词表（来自 WorldWeatherOnline.com）
WEATHER_CONDITION_MAP: Dict[str, str] = {
    # --- 雨 ---
    "Light drizzle": "小雨",
    "Patchy light drizzle": "零星小雨",
    "Patchy light rain": "零星小雨",
    "Light rain": "小雨",
    "Moderate rain at times": "中雨",
    "Moderate rain": "中雨",
    "Heavy rain at times": "大雨",
    "Heavy rain": "大雨",
    "Light rain shower": "阵雨",
    "Moderate or heavy rain shower": "强阵雨",
    "Torrential rain shower": "暴雨",
    "Rain": "雨",
    "Drizzle": "毛毛雨",
    # --- 雪 ---
    "Patchy light snow": "零星小雪",
    "Light snow": "小雪",
    "Patchy moderate snow": "中雪",
    "Moderate snow": "中雪",
    "Patchy heavy snow": "大雪",
    "Heavy snow": "大雪",
    "Light snow showers": "阵雪",
    "Moderate or heavy snow showers": "强阵雪",
    "Snow": "雪",
    "Blizzard": "暴风雪",
    "Blowing snow": "吹雪",
    # --- 雷暴/冰雨 ---
    "Patchy light rain with thunder": "雷阵雨",
    "Moderate or heavy rain with thunder": "雷阵雨伴有冰雹",  # 或强雷阵雨
    "Patchy light snow with thunder": "雷伴雪",
    "Moderate or heavy snow with thunder": "雷伴暴雪",
    "Thundery outbreaks possible": "可能有雷暴",
    "Thunder": "雷",
    "Ice pellets": "冰粒",
    "Light sleet": "雨夹雪",
    "Moderate or heavy sleet": "强雨夹雪",
    "Light sleet showers": "阵雨夹雪",
    "Moderate or heavy sleet showers": "强阵雨夹雪",
    "Freezing drizzle": "冻雨",
    "Freezing fog": "冻雾",
    # --- 雾/阴/多云/晴 ---
    "Mist": "薄雾",
    "Fog": "雾",
    "Overcast": "阴",
    "Cloudy": "多云",
    "Partly cloudy": "晴间多云",
    "Patchy rain possible": "可能有零星小雨",
    "Patchy snow possible": "可能有零星小雪",
    "Patchy sleet possible": "可能有雨夹雪",
    "Patchy freezing drizzle possible": "可能有冻雨",
    "Clear": "晴",
    "Sunny": "晴",
    # 兜底常见短词小写匹配（兼容各种大小写组合）
    "clear": "晴", "sunny": "晴",
    "clouds": "多云", "cloudy": "多云",
    "overcast": "阴",
    "rain": "雨", "drizzle": "毛毛雨",
    "snow": "雪",
    "fog": "雾", "mist": "薄雾",
    "thunderstorm": "雷阵雨", "thunder": "雷",
    "haze": "霾",
}

# ---- 风向 8/16 方位（英文缩写 -> 中文） ----
# https://www.worldweatheronline.com/weather-api/api/docs/windspeed16point.aspx
WIND_DIRECTION_MAP: Dict[str, str] = {
    # 8 方位（基础）
    "N": "北风", "NNE": "东北偏北风", "NE": "东北风", "ENE": "东北偏东风",
    "E": "东风", "ESE": "东南偏东风", "SE": "东南风", "SSE": "东南偏南风",
    "S": "南风", "SSW": "西南偏南风", "SW": "西南风", "WSW": "西南偏西风",
    "W": "西风", "WNW": "西北偏西风", "NW": "西北风", "NNW": "西北偏北风",
    # 常见单字母/二字母兜底（wttr.in 有时只返回 8 方位简化）
    "NB": "北风", "EB": "东风", "SB": "南风", "WB": "西风",
    "NORTHEAST": "东北风", "NORTHWEST": "西北风",
    "SOUTHEAST": "东南风", "SOUTHWEST": "西南风",
    "NORTH": "北风", "SOUTH": "南风", "EAST": "东风", "WEST": "西风",
}


# =============================================================================
# 二、【辅助函数：km/h -> 蒲福风力等级 0~12】
# =============================================================================

def kmh_to_beaufort(kmh_str: str) -> int:
    """
    风速 km/h -> 蒲福风力等级（0~12 级）。
    采用气象部门通用换算表；解析失败返回 0。
    """
    try:
        kmh = float(kmh_str)
    except (ValueError, TypeError):
        return 0
    if kmh < 1:
        return 0
    elif kmh <= 5:
        return 1
    elif kmh <= 11:
        return 2
    elif kmh <= 19:
        return 3
    elif kmh <= 28:
        return 4
    elif kmh <= 38:
        return 5
    elif kmh <= 49:
        return 6
    elif kmh <= 61:
        return 7
    elif kmh <= 74:
        return 8
    elif kmh <= 88:
        return 9
    elif kmh <= 102:
        return 10
    elif kmh <= 117:
        return 11
    else:
        return 12  # 12 级及以上（台风）


# =============================================================================
# 三、【翻译函数】
# =============================================================================

def translate_condition(raw_desc: str) -> str:
    """把 wttr.in 的英文天气现象翻译为简洁中文；找不到匹配时做规范化兜底。"""
    if not raw_desc:
        return "未知"
    key = raw_desc.strip()
    # 1) 完全匹配（大小写敏感的精确值）
    if key in WEATHER_CONDITION_MAP:
        return WEATHER_CONDITION_MAP[key]
    # 2) 首字母大写式：Light Drizzle → Light drizzle
    if key in WEATHER_CONDITION_MAP:
        return WEATHER_CONDITION_MAP[key]
    # 3) 全小写兜底（Clouds / Rain / Clear / Thunderstorm 等简单词）
    lower = key.lower()
    if lower in WEATHER_CONDITION_MAP:
        return WEATHER_CONDITION_MAP[lower]
    # 4) 关键词包含式（对付变体/拼接）
    if any(k in lower for k in ("thunder", "storm")):
        return "雷阵雨"
    if "drizzle" in lower:
        return "小雨"
    if any(k in lower for k in ("shower", "rain")):
        if "heavy" in lower:
            return "大雨"
        if "moderate" in lower:
            return "中雨"
        if "light" in lower or "patchy" in lower:
            return "零星小雨"
        return "雨"
    if "snow" in lower:
        if "heavy" in lower:
            return "大雪"
        if "moderate" in lower:
            return "中雪"
        if "light" in lower or "patchy" in lower:
            return "小雪"
        return "雪"
    if "sleet" in lower or ("rain" in lower and "snow" in lower):
        return "雨夹雪"
    if "fog" in lower:
        return "雾"
    if "mist" in lower or "haze" in lower:
        return "薄雾"
    if "clear" in lower or "sunny" in lower:
        return "晴"
    if "cloud" in lower:
        if "partly" in lower:
            return "晴间多云"
        if "overcast" in lower:
            return "阴"
        return "多云"
    if "overcast" in lower:
        return "阴"
    # 5) 实在翻译不出来：原样返回（但至少去掉多余空白），绝不丢字段
    return key.strip()


def translate_wind_dir(raw_dir: str) -> str:
    """风向英文缩写 -> 中文；解析失败返回空字符串。"""
    if not raw_dir:
        return ""
    key = raw_dir.strip()
    if key in WIND_DIRECTION_MAP:
        return WIND_DIRECTION_MAP[key]
    upper = key.upper()
    if upper in WIND_DIRECTION_MAP:
        return WIND_DIRECTION_MAP[upper]
    # 只取字母部分（兼容 "WNW 090" 这种带角度的返回）
    m = re.search(r'[NSEW]{1,3}', key)
    if m:
        return WIND_DIRECTION_MAP.get(m.group(1).upper(), "")
    return ""


# =============================================================================
# 四、【城市名规范化】
# =============================================================================

def normalize_city_name(requested_city: str, api_area_name: str) -> str:
    """
    城市名规范化：
      1) 优先使用用户请求的原始城市名（requested_city = extract_city 的结果，已经是干净中文）
      2) 如果 requested_city 为空，再从 API 返回值里做"中文清洗"（滤掉 Changshashih 这种拼音+市尾缀污染）
    """
    # Case A：用户明确说了城市名，直接用（永远是中文，如 "长沙"）
    if requested_city and re.search(r'[\u4e00-\u9fa5]', requested_city):
        clean = requested_city.strip()
        # 去掉尾缀"市"（可选，但通常播报"长沙天气"比"长沙市天气"更自然）；这里保留用户原词
        return clean

    # Case B：用户没说，尝试用 API 的 areaName 清洗
    if not api_area_name:
        return "本地"

    # 如果 areaName 本身就是中文（比如 areaName=[{"value":"长沙"}]），直接用
    if re.search(r'[\u4e00-\u9fa5]', api_area_name):
        return api_area_name.strip()

    # Case C：是英文拼音（Changshashih / Xianggang-shi / Beijing）→ 不播报拼音，直接回退 "本地"
    # （拼音翻译为中文需要额外词典，成本高且易错，对 TTS 不友好）
    return "本地"


# =============================================================================
# 五、【TTS 友好格式化】
# =============================================================================

def format_weather_report(
    city: str,
    condition_cn: str,
    temp_celsius: str,
    humidity: str,
    wind_dir_cn: str,
    beaufort_level: int,
    windspeed_kmh: str = "",
) -> str:
    """
    把 API 解析出的结构化字段，转为适合 TTS 播报的一句自然中文。

    模板（无标点歧义、主谓宾顺序符合口语）：
      "今天[城市]天气[有/是][现象]，当前温度 [温度] 度，相对湿度 [湿度]%，[风向] [X] 级。"

    细节：
      - 天气现象前加适当动词：
          雨/雪/雷阵雨/雨夹雪/雾 → "有小雨"/"有小雪"/"有雷阵雨"
          晴/多云/晴间多云/阴 → "天气晴"/"天气多云"
      - 湿度前加 "相对" 两个字，TTS 停顿更自然
      - 风力 0 级（无风）→ 只说"无风"，不说"北风 0 级"
      - 风向未知但有风速 → "风力 X 级"
      - 风向+风速 → "西北风 3 级"
    """
    # 1) 天气现象自然化拼接
    rain_like = {"雨", "小雨", "中雨", "大雨", "暴雨", "毛毛雨", "阵雨", "强阵雨",
                 "雪", "小雪", "中雪", "大雪", "阵雪", "强阵雪", "暴风雪", "吹雪",
                 "雷阵雨", "强雷阵雨", "雷阵雨伴有冰雹", "雷伴雪", "雷伴暴雪",
                 "雨夹雪", "强雨夹雪", "阵雨夹雪", "强阵雨夹雪",
                 "冻雨", "雾", "冻雾", "薄雾", "霾", "冰粒"}
    if condition_cn in rain_like:
        weather_part = f"天气有{condition_cn}"
    else:
        weather_part = f"天气{condition_cn}"

    # 2) 温度
    try:
        temp_int = int(float(temp_celsius))
        temp_part = f"当前温度 {temp_int} 度"
    except (ValueError, TypeError):
        temp_part = f"当前温度 {temp_celsius} 度"

    # 3) 湿度
    hum_part = f"相对湿度 {humidity}%"

    # 4) 风（最复杂：0级 / 有风向 / 只有风速 三种情况）
    if beaufort_level <= 0:
        wind_part = "无风"
    elif wind_dir_cn:
        wind_part = f"{wind_dir_cn} {beaufort_level} 级"
    else:
        wind_part = f"风力 {beaufort_level} 级"

    # 5) 最终拼装："今天 长沙 天气有小雨，当前温度 23 度，相对湿度 94%，西北风 13 级。"
    report = f"今天{city}{weather_part}，{temp_part}，{hum_part}，{wind_part}。"
    # 收尾小清理：避免"今天本地天气..."这种有点怪的说法（如果是本地可以不加"本地"，看用户习惯；这里保留即可）
    # 去掉多余标点：双重逗号、前后空格等
    report = re.sub(r'，+', '，', report).strip('，, .。') + "。"
    return report


# =============================================================================
# 六、【对外主 API：get_weather / extract_city】
# =============================================================================

async def get_weather(city: str) -> Optional[Dict[str, str]]:
    """
    查询指定城市的实时天气。

    与旧版的关键区别：
      - description 不再直接拼 API 返回的拼音/英文，而是走 format_weather_report
      - city 字段：使用请求传入的 city（已经是 extract_city 解析的干净中文），不用 API 返回的 areaName

    Args:
        city: 城市名，已经是 extract_city() 返回的干净中文名，如 "长沙"、"北京"

    Returns:
        {
            "city": "长沙",                 # 规范化后的中文城市名
            "condition": "小雨",             # 中文化天气现象
            "temperature": "23°C",           # 温度（字符串，带单位）
            "humidity": "94%",               # 湿度（字符串，带单位）
            "wind": "西北风 13 级",           # 风向+蒲福风级中文
            "beaufort": "13",                # 蒲福风力等级数字串（便于其他模块读）
            "wind_dir": "西北风",            # 中文风向
            "windspeed_kmh": "? 或原始数字",  # 原始风速 km/h（保留用于调试）
            "description": "今天长沙天气有小雨，当前温度 23 度，相对湿度 94%，西北风 13 级。",
        }
        或 None（查询失败时）
    """
    import httpx

    if not city:
        return None

    try:
        # 使用 wttr.in 免费 API（无需 API Key）
        # note: 这里传入 city（中文），wttr.in 通常能识别；失败时会走异常分支
        url = f"https://wttr.in/{city}?format=j1"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        # 解析 JSON
        current = data.get("current_condition", [])
        if not current:
            return None

        cur = current[0]
        temp_c = cur.get("temp_C", "?")
        humidity = cur.get("humidity", "?")
        desc_raw = cur.get("weatherDesc", [{}])[0].get("value", "未知")
        windspeed_kmh = cur.get("windspeedKmph", "?")
        wind_dir_raw = cur.get("winddir16Point", "")

        # 【修复1】城市名：一律用请求里的 city（extract_city 解析的干净中文），
        #          绝不使用 API 返回的 Changshashih 这种拼音拼接值。
        api_area_name = ""
        try:
            nearest = data.get("nearest_area", [{}])[0] or {}
            area_list = nearest.get("areaName", [{}]) or [{}]
            api_area_name = area_list[0].get("value", "")
        except (IndexError, KeyError, TypeError):
            api_area_name = ""
        final_city = normalize_city_name(city, api_area_name)

        # 【修复2】天气现象 + 风向 中文化
        condition_cn = translate_condition(desc_raw)
        wind_dir_cn = translate_wind_dir(wind_dir_raw)

        # 【修复3】风速 km/h -> 蒲福风力等级（1~12 级）
        beaufort = kmh_to_beaufort(windspeed_kmh)

        # 【修复4】TTS 友好格式化
        description = format_weather_report(
            city=final_city,
            condition_cn=condition_cn,
            temp_celsius=temp_c,
            humidity=humidity,
            wind_dir_cn=wind_dir_cn,
            beaufort_level=beaufort,
            windspeed_kmh=str(windspeed_kmh),
        )

        # wind 字段：蒲福风力+风向（给下游需要单独展示时用）
        if beaufort <= 0:
            wind_txt = "无风"
        elif wind_dir_cn:
            wind_txt = f"{wind_dir_cn} {beaufort} 级"
        else:
            wind_txt = f"风力 {beaufort} 级"

        return {
            "city": final_city,
            "condition": condition_cn,
            "temperature": f"{temp_c}°C" if str(temp_c) != "?" else "?°C",
            "humidity": f"{humidity}%" if str(humidity) != "?" else "?%",
            "wind": wind_txt,
            "beaufort": str(beaufort),
            "wind_dir": wind_dir_cn,
            "windspeed_kmh": str(windspeed_kmh),
            "description": description,
        }

    except httpx.HTTPError as e:
        print(f"[Weather] HTTP 请求失败: {e}")
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"[Weather] 数据解析失败: {e}")
        return None
    except Exception as e:
        print(f"[Weather] 未知错误: {e}")
        return None


def extract_city(text: str) -> Optional[str]:
    """
    从用户输入文本中提取城市名（保持为干净的中文名，供 normalize_city_name / get_weather 第一优先级使用）。

    支持的句式：
    - "今天长沙天气怎么样" → "长沙"
    - "北京天气" → "北京"
    - "查一下上海的天气" → "上海"
    - "明天广州天气如何" → "广州"
    - 默认城市配置（未指定城市时，返回预设城市）
    """
    # 常见城市列表（按长度倒序，优先匹配长城市名）
    cities = [
        "石家庄", "哈尔滨", "长春", "呼和浩特", "乌鲁木齐",
        "西宁", "银川", "兰州", "西安", "拉萨",
        "贵阳", "成都", "重庆", "昆明", "南宁",
        "广州", "深圳", "珠海", "东莞", "佛山",
        "长沙", "武汉", "郑州", "济南", "青岛",
        "南京", "苏州", "无锡", "徐州", "常州",
        "合肥", "杭州", "宁波", "温州", "福州",
        "厦门", "南昌", "台北", "香港", "澳门",
        "北京", "上海", "天津",
    ]

    cities_sorted = sorted(cities, key=len, reverse=True)
    for city in cities_sorted:
        if city in text:
            return city

    # 尝试提取不在地名单中的 2~4 字中文城市名
    # 句式：今天/明天/后天 + （中文2~4字） + 天气/气温
    m = re.search(
        r'(?:今天|明天|后天|查一下|查查|看看|请问|帮我)?\s*'
        r'([\u4e00-\u9fa5]{2,4}?)\s*'
        r'(?:的|地)?\s*(?:天气|气温|温度|下雨|降水|刮风)',
        text,
    )
    if m:
        candidate = m.group(1)
        skip_words = {"什么", "怎么样", "如何", "怎样", "这个", "那个",
                      "咱们", "你们", "他们", "我们", "本地", "全国",
                      "现在", "今天", "明天", "后天", "昨儿", "昨日"}
        if candidate and candidate not in skip_words:
            # 再加一道：必须是 2~4 个汉字（绝大多数地级市名的长度）
            if 2 <= len(candidate) <= 4 and re.fullmatch(r'[\u4e00-\u9fa5]+', candidate):
                return candidate

    return None
