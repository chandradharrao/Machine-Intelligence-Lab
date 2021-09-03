class X:
    def __init__(self):
        self.y=0
    
    def inc(self,i):
        i+=1
        print(f"Inc i to {i}")

    def do(self):
        self.inc(self.y)
        print(self.y)

x=X()
x.do()
