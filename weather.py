import json
import os

import requests
from langchain_core.tools import tool
from loguru import logger
from pypinyin import lazy_pinyin


@tool
@logger.catch
def get_weather(location: str):
    """
    调用 Weather API 查询指定位置今天的天气数据。

    参数:
        location: 查询的地点的中文名，例如 "北京" 或 "成都"。

    返回:
        API 返回的 JSON 数据，若调用失败则抛出异常。
    """
    location = location.replace("市", "").replace("县", "")
    location = "".join(lazy_pinyin(location))
    logger.info(f"###天气API正在查询 {location}")
    api_key = os.getenv('VISUAL_CROSSING_API_KEY')
    if not api_key:
        raise ValueError("请设置环境变量 VISUAL_CROSSING_API_KEY 以提供 Visual Crossing API Key")

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location},CN/today?lang=zh&include=days&unitGroup=metric"
    params = {
        "key": api_key,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        # logger.info(f"API返回 {json.dumps(response.json(), indent=2)}")
        return process_weather_data(response.json())
    else:
        # response.raise_for_status()
        return "地区名称有误,请换一个中文名.要求的名称为中国境内的市或县的名称.如果你不清楚用户询问的地区属于哪个市或县请询问用户.2次失败请告知用户'无法查询到xx的天气',严禁编造数据"


def process_weather_data(weather_json):
    # 提取核心信息
    location = weather_json.get("resolvedAddress", "未知地点")
    description = weather_json.get("days", [{}])[0].get("description", "无天气描述")
    conditions = weather_json.get("days", [{}])[0].get("conditions", "无天气条件")
    temp_max = weather_json.get("days", [{}])[0].get("tempmax", "未知")
    temp_min = weather_json.get("days", [{}])[0].get("tempmin", "未知")
    temp_current = weather_json.get("days", [{}])[0].get("temp", "未知")
    feels_like = weather_json.get("days", [{}])[0].get("feelslike", "未知")
    humidity = weather_json.get("days", [{}])[0].get("humidity", "未知")
    wind_speed = weather_json.get("days", [{}])[0].get("windspeed", "未知")
    wind_dir = weather_json.get("days", [{}])[0].get("winddir", "未知")
    uv_index = weather_json.get("days", [{}])[0].get("uvindex", "未知")
    sunrise = weather_json.get("days", [{}])[0].get("sunrise", "未知")
    sunset = weather_json.get("days", [{}])[0].get("sunset", "未知")
    visibility = weather_json.get("days", [{}])[0].get("visibility", "未知")
    pressure = weather_json.get("days", [{}])[0].get("pressure", "未知")
    precip = weather_json.get("days", [{}])[0].get("precip", "未知")

    # 构建自然语言描述
    weather_report = (
        f"在{location}，当前天气情况：{description}。"
        f"今日最高温度为{temp_max}°C，最低温度为{temp_min}°C，"
        f"当前温度为{temp_current}°C，体感温度为{feels_like}°C。"
        f"空气湿度为{humidity}% ，风速为{wind_speed} km/h，风向为{wind_dir}°。"
        f"紫外线指数为{uv_index}，可见度为{visibility} km，气压为{pressure} hPa。"
        f"天气状况：{conditions}。日出时间为{sunrise}，日落时间为{sunset}。"
        f"预计降水量{precip}mm/ml。"
    )

    return weather_report


if __name__ == '__main__':
    # 示例数据
    weather_json = {
        "queryCost": 1,
        "latitude": 30.2724,
        "longitude": 120.206,
        "resolvedAddress": "中国杭州",
        "address": "hangzhou,CN",
        "timezone": "Asia/Shanghai",
        "tzoffset": 8.0,
        "days": [
            {
                "datetime": "2025-04-13",
                "datetimeEpoch": 1744473600,
                "tempmax": 22.0,
                "tempmin": 7.0,
                "temp": 15.1,
                "feelslikemax": 22.0,
                "feelslikemin": 4.3,
                "feelslike": 14.5,
                "dew": -0.5,
                "humidity": 35.9,
                "precip": 0.0,
                "precipprob": 0.0,
                "precipcover": 0.0,
                "preciptype": None,
                "snow": 0.0,
                "snowdepth": 0.0,
                "windgust": 51.8,
                "windspeed": 25.2,
                "winddir": 240.3,
                "pressure": 1011.8,
                "cloudcover": 24.6,
                "visibility": 16.5,
                "solarradiation": 319.1,
                "solarenergy": 27.8,
                "uvindex": 10.0,
                "severerisk": 10.0,
                "sunrise": "05:34:34",
                "sunriseEpoch": 1744493674,
                "sunset": "18:25:27",
                "sunsetEpoch": 1744539927,
                "moonphase": 0.5,
                "conditions": "部分多云",
                "description": "阳光明媚的下午.",
                "icon": "partly-cloudy-day",
                "stations": [
                    "ZSHC"
                ],
                "source": "comb"
            }
        ],
        "stations": {
            "ZSHC": {
                "distance": 22317.0,
                "latitude": 30.22,
                "longitude": 120.43,
                "useCount": 0,
                "id": "ZSHC",
                "name": "ZSHC",
                "quality": 50,
                "contribution": 0.0
            }
        }
    }
    weather_report = process_weather_data(weather_json)
    print(weather_report)
