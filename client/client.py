#!/usr/bin/env python3
import asyncio
import aiohttp
import argparse
import os
import io

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to image to recognize", required=True)
args = vars(ap.parse_args())

HOST = os.getenv('HOST', 'cosmetic_recognition_server')
PORT = os.getenv('PORT', '7070')


async def get_photo_bytes_by_url():
    async with aiohttp.ClientSession() as session:
        async with session.get(args['image']) as resp:
            image_bytes = await resp.read()
    return image_bytes


def get_photo_bytes_local():
    with open(args['image'], 'rb') as f:
        image_bytes = io.BytesIO(f.read())
    return image_bytes


async def main():
    if args['image'].startswith('http'):
        image = await get_photo_bytes_by_url()
    else:
        image = get_photo_bytes_local()

    async with aiohttp.ClientSession() as session:
        async with session.post(url='http://' + HOST + ':' + str(PORT) + "/image", data={'image': image}) as resp:
            response = await resp.text()
            print(response)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
