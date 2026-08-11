"""
天气查询服务 - 免费天气 API 集成

使用 wttr.in 免费服务，无需 API Key：
  - 全球城市支持
  - 返回 JSON 格式的实时天气数据
  - 无调用次数限制

如果 wttr.in 在国内网络不可用，可配置备用 API。
"""

import json
import re
from typing import Dict, Optional


async def get_weather(city: str) -> Optional[Dict[str, str]]:
    """
    查询指定城市的实时天气。

    Args:
        city: 城市名，如 "长沙"、"北京"

    Returns:
        {
            "city": "长沙",
            "condition": "晴",
            "temperature": "28°C",
            "humidity": "65%",
            "wind": "3级",
            "description": "长沙今天天气：晴，28°C，湿度65%，风力3级"
        }
        或 None（查询失败时）
    """
    import httpx

    if not city:
        return None

    try:
        # 使用 wttr.in 免费 API（无需 API Key）
        url = f"https://wttr.in/{city}?format=j1"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        # 解析返回的 JSON
        current = data.get("current_condition", [])
        if not current:
            return None

        cur = current[0]
        temp = cur.get("temp_C", "?")
        humidity = cur.get("humidity", "?")
        desc = cur.get("weatherDesc", [{}])[0].get("value", "未知")
        wind_speed = cur.get("windspeedKmph", "?")
        wind_dir = cur.get("winddir16Point", "")

        # 获取城市中文名（wttr.in 返回的 areaName 可能已中文化）
        area_name = city
        try:
            nearest = data.get("nearest_area", [{}])[0]
            area_list = nearest.get("areaName", [{}])
            if area_list and area_list[0].get("value"):
                area_name = area_list[0]["value"]
        except (IndexError, KeyError, TypeError):
            pass

        # 构建自然描述
        description = f"{area_name}今天天气：{desc}，{temp}°C，湿度{humidity}%，风力{wind_speed}级{wind_dir}"

        return {
            "city": area_name,
            "condition": desc,
            "temperature": f"{temp}°C",
            "humidity": f"{humidity}%",
            "wind": f"{wind_speed}级",
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
    从用户输入文本中提取城市名。

    支持的句式：
    - "今天长沙天气怎么样" → "长沙"
    - "北京天气" → "北京"
    - "查一下上海的天气" → "上海"
    - "明天广州天气如何" → "广州"
    - 默认城市配置（未指定城市时，返回预设城市）

    Returns:
        城市名，或 None（无法识别且无默认城市）
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

    # 按长度倒序排序，长城市名优先匹配
    cities_sorted = sorted(cities, key=len, reverse=True)
    for city in cities_sorted:
        if city in text:
            return city

    # 尝试提取不在地名单中的城市名
    # 句式：今天/明天/后天 + <任意词> + 天气
    m = re.search(r'(今天|明天|后天|查一下|查查)?\s*(\w{2,4}?)\s*(天气|气温)', text)
    if m:
        candidate = m.group(2)
        # 过滤掉非城市词
        skip_words = {"什么", "怎么样", "如何", "怎样", "这个", "那个", "咱们", "你们", "他们", "我们"}
        if candidate and candidate not in skip_words:
            return candidate

    return None