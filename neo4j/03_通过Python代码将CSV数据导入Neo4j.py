# -*- coding: utf-8 -*-
# @Time    : 2025/12/30 16:57
# @Author  : 老冰棍
# @File    : 03_通过Python代码将CSV数据导入Neo4j.py
# @Software: PyCharm
import pandas as pd

if __name__ == '__main__':
    from py2neo import Graph

    profile = "bolt://8.148.255.196:7687"
    name = "neo4j"
    auth = ("neo4j", "12345678")

    graph = Graph(profile=profile, name=name, auth=auth)

    df = pd.read_csv(r"artists3.csv", header=None, sep=",")

    cql_str = """
        MERGE (a:Artist{id:toInteger($id)})
        ON CREATE SET
            a.id = toInteger($id),
            a.name = $name,
            a.year = toInteger($year),
            a.age = toInteger($age),
            a.created = timestamp()
        ON MATCH SET
            a.year = toInteger($year),
            a.age = toInteger($age)
        RETURN a;
    """

    for line in df.values:
        r1 = graph.run(cql_str, id =line[0], name = line[1], year = line[2], age = line[3])
        print(r1.data())