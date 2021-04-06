import asyncio
import search
import json
import logging
import config
from aiohttp import web
from typing import Awaitable, Callable
import time

router = web.RouteTableDef()
logging.basicConfig(filename=config.PATH_TO_LOG_FILE,
                    format='%(asctime)s [%(levelname)s] - %(message)s',
                    level=logging.INFO)


def handle_json_error(
    func: Callable[[web.Request], Awaitable[web.Response]]
) -> Callable[[web.Request], Awaitable[web.Response]]:
    async def handler(request: web.Request) -> web.Response:
        try:
            return await func(request)
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            logging.error(ex)
            return web.json_response(
                {"status": "failed", "reason": str(ex)}, status=400
            )

    return handler


@router.post("/image")
@handle_json_error
async def get_product_by_image(request: web.Request) -> web.Response:
    data = await request.post()
    try:
        t1 = time.time()
        json_result = search.perform_search(image_bytes=data['image'].file.read())
        t2 = time.time()
        print(t2 - t1)
        return web.json_response(json_result, dumps=json_dumps_utf)
    except Exception as ex:
        logging.error(ex)


def json_dumps_utf(data):
    return json.dumps(data, ensure_ascii=False)


async def init_app() -> web.Application:
    app = web.Application()
    app.add_routes(router)
    return app

if __name__ == '__main__':
    web.run_app(init_app(), port=7070)