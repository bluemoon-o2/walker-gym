from gym import Point

a = Point(m=1.0, pos=[0, 0, 0], v=[0, 0, 0])
b = Point(m=1.0, pos=[10, 10, 10], v=[0, 0, 0])
c = Point(m=1.0, pos=[10, 0, 10], v=[0, 0, 0])

print(a)
print(b)

Point.play(k=50)
