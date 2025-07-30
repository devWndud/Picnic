import requests
import urllib.parse

def tmap_pedestrian_route(api_key, start, end, search_option="10", sort="custom"):
    url = "https://apis.openapi.sk.com/tmap/routes/pedestrian?version=1"
    headers = {
        "appKey": api_key,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    # 출발지/목적지 좌표와 명칭
    startX, startY, startName = start
    endX, endY, endName = end

    # 출발지/목적지 이름 URL 인코딩 (UTF-8)
    startName_enc = urllib.parse.quote(startName)
    endName_enc = urllib.parse.quote(endName)

    payload = {
        "startX": str(startX),
        "startY": str(startY),
        "endX": str(endX),
        "endY": str(endY),
        "startName": startName_enc,
        "endName": endName_enc,
        "searchOption": search_option,
        "sort": sort
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()

    features = data.get("features", [])
    for f in features:
        props = f.get("properties", {})
        geom = f.get("geometry", {})
        print(f"Type: {f.get('type')}, Geometry Type: {geom.get('type')}")
        print(f"Description: {props.get('description', '')}")
        if geom.get("type") == "Point":
            coords = geom.get("coordinates", [])
            print(f"Coordinates: {coords}")
            print(f"Point Type: {props.get('pointType', '')}")
        elif geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            print(f"Line length: {len(coords)} points")
            print(f"Distance: {props.get('distance', '')}m, Time: {props.get('time', '')}s")
        print("-" * 40)

if __name__ == "__main__":
    API_KEY = "UdkX1DkUFe3gGmck3iXH81Dw4fURyDjYaYFfvuem"
    # 예: 제주 올레 2코스 출발지 광치기해변 → 온평포구

    start = (127.118784, 36.801487, "광치기해변")
    end = (127.110363, 36.802439, "온평포구")

    tmap_pedestrian_route(API_KEY, start, end)
