import asyncio

import httpx
import runpod


async def handler(job):
    job_input = job["input"]
    async with httpx.AsyncClient() as client:
        client.timeout = httpx.Timeout(600)
        is_img2img = job_input["is_img2img"]
        if is_img2img:
            endpoint = "http://localhost:7861/sdapi/v1/img2img"
        else:
            endpoint = "http://localhost:7861/sdapi/v1/txt2img"
        payload = job_input["payload"]
        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.content.decode("utf-8")
        chunk_size = 1_000_000
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]


runpod.serverless.start(
    {
        "handler": handler,
        "return_aggregate_stream": True,
    }
)