import time

class MemoryLifecycle:

    def score(self, memory):
        """
        计算记忆重要性
        """

        score = 0

        # 出现关键词增强
        if "error" in memory:
            score += 3

        if "GPU" in memory:
            score += 2

        # 时间衰减
        age = time.time() - memory.get("ts", time.time())
        score -= age / 86400  # 天级衰减

        return score

    def is_valid(self, memory):

        return self.score(memory) > 0