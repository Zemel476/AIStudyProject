# -*- coding: utf-8 -*-
# @Time    : 2025/12/30 15:10
# @Author  : 老冰棍
# @File    : 01_py2neo操作api.py
# @Software: PyCharm
if __name__ == '__main__':
    from py2neo import Graph
    from py2neo.cypher import Cursor

    profile = "bolt://8.148.255.196:7687" # 远程图数据库地址
    name = "neo4j"  # 图数据库database名称

    auth = ("neo4j", "12345678") # 用户名和密码
    graph = Graph(profile=profile, name=name, auth=auth)

    # 参加数据
    # r1: Cursor = graph.run(
    #     """
    #         CREATE (b:Book {name:$name, price:$book_price}) RETURN b
    #     """,
    #     name='深度学习基础',
    #     book_price=13.52
    # )
    #
    # print(type(r1))
    # print(r1)
    # print(r1.data())

    # RUN方法执行结果是一个迭代器
    # r2 = graph.run("MATCH (b:Book {name:$name}) RETURN b.name AS name, b.price AS price", name='深度学习基础')
    # print(r2.to_data_frame())

    print("=" * 30)

    # 数据迭代
    r3 = graph.run("MATCH (n) RETURN n")
    while r3.forward():
        current_forward = r3.current
        print(type(current_forward))
        print(current_forward)
        print(current_forward['n'])
        print(current_forward['n'].get('name'))
        print(current_forward['n'].get('price'))
        print("=" * 30)