import asyncio

import httpx
import runpod


async def wait_for_sdapi_ready():
    while True:
        try:
            endpoint = "http://localhost:7861/sdapi/v1/memory"
            async with httpx.AsyncClient() as client:
                response = await client.get(endpoint)
                response.raise_for_status()
                return
        except Exception:
            await asyncio.sleep(0)


async def handler(job):
    await wait_for_sdapi_ready()
    job_input = job["input"]
    endpoint = "http://localhost:7861/sdapi/v1/txt2img"
    async with httpx.AsyncClient() as client:
        client.timeout = httpx.Timeout(None)
        response = await client.post(endpoint, json=job_input)
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