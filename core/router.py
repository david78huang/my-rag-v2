class MemoryRouter:

    def route(self, extracted):

        facts = {}
        semantic = []
        graph = []

        for k, v in extracted.get("facts", {}).items():
            facts[k] = v

        for item in extracted.get("semantic", []):
            semantic.append(item)

        for s, r, t in extracted.get("graph", []):
            graph.append((s, r, t))

        return facts, semantic, graph