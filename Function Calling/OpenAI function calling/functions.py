import json


def get_current_weather(location: str, unit: str):
    """
    현재 지역의 날씨를 갖고 오는 함수

    Args:
        location: 지역
        unit: 온도 표기 방식

    Returns:
        json: 날씨 정보
    """
    weather_info = {
        "location": location,
        "temperature": "24", # 이 함수가 제대로 불러와졌는지 확인하기 위해 온도를 고정
        "unit": unit,
        "forecast": ["sunny", "windy"]
    }

    return json.dumps(weather_info, ensure_ascii=False)


def get_location_nickname(location: str):
    """
    지역 이름에 어울리는 별명을 생성하는 함수

    Args:
        location: 지역 이름

    Returns:
        json: 지역 이름과 별명을 포함한 JSON 형식의 문자열
    """

    nickanme = "선풍기" # 이 함수가 제대로 불러와졌는지 확인하기 위해 별명 고정

    result = {
        "location": location,
        "nickname": nickanme
    }

    return json.dumps(result, ensure_ascii=False)