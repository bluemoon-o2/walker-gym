import random
from .engine import Point


def damp(p: Point, k: float):
    p.forced(-k * p.v)


class Environment:
    def __init__(self,creaturelist,in3d=False,g=100,dampk=0,groundhigh=0,groundk=1000,grounddamp=100,friction=100,randsigma=0.1):
        self.creatures=creaturelist
        self.g=g
        self.in3d=in3d
        self.dampk=dampk
        self.ground=groundhigh
        self.groundk=groundk
        self.grounddamp=grounddamp
        self.friction=friction
        self.sigma=randsigma

        for i in self.creatures:
            for j in i.phys:
                j.v[0] += random.gauss(0, self.sigma)
                j.v[1] += random.gauss(0, self.sigma)
                if self.in3d:
                    j.v[2] += random.gauss(0, self.sigma)

    def run(self):
        for c in self.creatures:
            c.run1()
            for p in c.phys:
                p.forced([0, -self.g, 0])
                damp(p,self.dampk)

                if p.p[1]-self.ground<0:
                    p.color="red"
                    p.r=3
                    deep=(p.p[1]-self.ground)
                    p.forced([0, -self.groundk * deep, 0])
                    p.forced([0, -self.grounddamp * p.v[1], 0])
                    p.forced([p.v[0] * deep * self.friction, 0, p.v[2] * deep * self.friction])

                    # p.v=[0,p.v[1],0]
                else:
                    p.color="black"
                    p.r=1

    def step(self,t):
        self.run()
        Point.run1(t)