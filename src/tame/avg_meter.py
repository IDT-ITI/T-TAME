from queue import Queue


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, type: str = "ema", a: float = 0.001, k: int = 100):
        """_summary_

        Args:
            type (str, optional): options: "ema" for exponential moving average
              "mavg" for simple moving average and "avg" for average. Defaults to "ema".
            a (float, optional): _description_. Defaults to 0.001.
            k (int, optional): _description_. Defaults to 100.
        """
        self.type = type
        if self.type == "mavg":
            self.values: Queue[float] = Queue(maxsize=k)
        self.a = a
        self.k = k
        self.reset()

    def reset(self):
        self.val: float = 0.0
        self.avg: float = 0.0
        self.count: int = 0
        self.init = True

    def update(self, val: float):
        self.val = val
        if self.init:
            self.avg = self.val
            self.init = False
        else:
            if self.type == "ema":
                self.avg = self.a * self.val + (1 - self.a) * self.avg
            elif self.type != "avg":
                if not self.values.full():
                    self.values.put_nowait(self.val)
                    self.avg = self.val
                else:
                    last_val = self.values.get_nowait()
                    self.values.put_nowait(self.val)
                    self.avg = self.avg + (self.val - last_val) / self.k
            else:
                self.val = val
                self.avg = self.avg * (self.count / (self.count + 1)) + self.val / (
                    self.count + 1
                )
                self.count += 1

    def __call__(self) -> float:
        return self.avg

    def __str__(self) -> str:
        return str(self.avg)
