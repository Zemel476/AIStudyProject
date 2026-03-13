# -*- coding: utf-8 -*-
# @Time    : 2026/1/13 17:33
# @Author  : 老冰棍
# @File    : http_client_with_fastapi.py
# @Software: PyCharm

import requests

target_url = "http://127.0.0.1:9001"

def get_call(url, params):
    response = requests.get(url, json=params)

    if response.status_code == 200:
        print(f"服务器返回结果为: {response.json()}")
    else:
        print(f"客户端和服务器之间的连接存在问题: {response.status_code}")


def post_call(url, params, with_json=False):
    if with_json:
        response = requests.post(url, json=params)
    else:
        response = requests.post(url, data=params)

    if response.status_code == 200:
        print(f"服务器返回结果为: {response.json()}")
    else:
        print(f"客户端和服务器之间的连接存在问题: {response.status_code}")


def tt01():
    get_call(
        url=fr"{target_url}",
        params={}
    )

    get_call(
        url=fr"{target_url}/predict",
        params={
            'text': '请给我放一个王宝强的电影的幕后花絮',
            'topk': 3
        }
    )

    post_call(
        url=fr"{target_url}/predict",
        params={
            'text': '帮我打开一下空调',
            'topk': 3
        },
        with_json=True
    )

    post_call(
        url=fr"{target_url}/predict",
        params={
            'text': '记得明天5点提醒我戴耳机',
            'topk': 4
        },
        with_json=True
    )


if __name__ == '__main__':
    tt01()