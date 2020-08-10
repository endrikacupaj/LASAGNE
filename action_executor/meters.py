# meter class for storing results
class AccuracyMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.wrong = 0
        self.accuracy = 0

    def update(self, gold, result):
        if gold == result:
            self.correct += 1
        else:
            self.wrong += 1

        self.accuracy = self.correct / (self.correct + self.wrong)

class F1scoreMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0

    def update(self, gold, result):
        self.tp += len(result.intersection(gold))
        self.fp += len(result.difference(gold))
        self.fn += len(gold.difference(result))
        if self.tp > 0 or self.fp > 0:
            self.precision = self.tp / (self.tp + self.fp)
        if self.tp > 0 or self.fn > 0:
            self.recall = self.tp / (self.tp + self.fn)
        if self.precision > 0 or self.recall > 0:
            self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)
