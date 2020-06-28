from multiprocessing.reduction import ForkingPickler

class ForkingPickler4(ForkingPickler):

    @classmethod
    def dumps(cls, obj, protocol=4):
        return super().dumps(obj, protocol)

def dump(obj, file, protocol=4):
    ForkingPickler4(file, protocol).dump(obj)
