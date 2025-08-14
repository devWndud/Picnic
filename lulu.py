import asyncio, websockets, json
async def main():
    async with websockets.connect("ws://222.231.24.5:8001") as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            print(data["stream_id"], len(data["detections"]), data.get("times"))
asyncio.run(main())