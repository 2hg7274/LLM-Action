import json


def get_current_weather(location: str, unit: str):
    """
    지정된 지역(location)의 실시간 기상 정보를 조회하여, 요청된 온도 단위(unit)에 맞춰 JSON 형태로 반환합니다.

    Args:
        location: 조회를 원하는 지역(도시 이름, 행정구역 등)을 나타내는 문자열입니다. (예: "서울", "부산", "New York", "Tokyo" 등)
        unit: 온도 표기 방식을 결정하는 문자열입니다. (예: "화씨(Fahrenheit)", "섭씨(Celsius)" / 기본 단위는 "섭씨(Celsius)" 입니다.)

    Returns:
        json: 해당 지역의 날씨 정보를 담은 JSON 객체를 반환합니다.
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
    입력으로 주어진 지역 이름(location)을 바탕으로, 해당 지역에 어울리는 독특하거나 재미있는 별명을 생성합니다.
    생성된 별명과 원래 지역 이름을 함께 JSON 형식으로 반환합니다.

    Args:
        location: 별명을 생성하고자 하는 지역(도시, 마을, 국가 등)의 이름입니다.

    Returns:
        json: 생성된 별명과 원본 지역 이름을 담은 JSON 객체를 반환합니다.
    """

    nickanme = "선풍기" # 이 함수가 제대로 불러와졌는지 확인하기 위해 별명 고정

    result = {
        "location": location,
        "nickname": nickanme
    }

    return json.dumps(result, ensure_ascii=False)