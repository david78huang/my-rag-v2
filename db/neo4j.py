from neo4j import GraphDatabase

class GraphMemory:

    def __init__(self):
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )

    def add(self, s, r, t):

        q = """
        MERGE (a:Entity {name:$s})
        MERGE (b:Entity {name:$t})
        MERGE (a)-[:REL {type:$r}]->(b)
        """

        with self.driver.session() as sess:
            sess.run(q, s=s, r=r, t=t)

    def query(self, entity):

        q = """
        MATCH (a)-[r]->(b)
        WHERE a.name=$e
        RETURN a.name, r.type, b.name
        """

        with self.driver.session() as sess:
            return list(sess.run(q, e=entity))