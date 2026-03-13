# -*- coding: utf-8 -*-
# @Time    : 2025/12/30 15:34
# @Author  : 老冰棍
# @File    : 02_neo4j driver 操作API.py
# @Software: PyCharm

from neo4j import GraphDatabase, Driver, RoutingControl


def add_friends(driver: Driver, db: str, name_1: str, name_2: str):
    query = """
        MERGE (a:Person {name: $name_1})
        MERGE (b:Person {name: $name_2})
        MERGE (a) -[:FRIEND]->(b)
        MERGE (b) -[:FRIEND]->(a)
    """
    driver.execute_query(query, name_1=name_1, name_2=name_2, database=db)




def print_friends(driver: Driver, db: str, name: str):
    query = """
        MATCH (a:Person) -[:FRIEND]-> (b:Person)
        WHERE a.name = $name
        RETURN b.name as name
        ORDER BY b.name
    """
    records, summary, keys = driver.execute_query(query, name=name, database=db, routing=RoutingControl.READ)
    for record in records:
        print(type(record))
        print(record)
        print(record['name'])


if __name__ == '__main__':
    url = "bolt://8.148.255.196:7687"
    db = "neo4j"
    auth = ("neo4j", "12345678")

    with GraphDatabase.driver(url, auth=auth) as driver:
        add_friends(driver, db, "小明", "小红")
        add_friends(driver, db, "小明", "小华")
        add_friends(driver, db, "小明", "小沪")
        add_friends(driver, db, "小华", "张三")
        add_friends(driver, db, "张三", "王五")
        print_friends(driver, db, "小明")