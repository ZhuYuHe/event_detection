class A(object):
    def __init__(self, config):
        self.config = config
    
    def modify(self):
        self.config[0] = 'modified'

class B(A):
    def __init__(self, config):
        self.config = config
        super(B, self).__init__(config)
    
    def test(self):
        self.modify()
    
    def getconfig(self):
        return self.config
        

def main():
    config = [1,2,3]
    b = B(config)
    b.test()
    print(b.getconfig())

if __name__ == '__main__':
    main()