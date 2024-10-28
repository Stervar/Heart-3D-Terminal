# Heart

# Heart

```python
import turtle
import math

def heart():
    turtle.speed(0)
    turtle.bgcolor("black")
    turtle.pensize(2)
    turtle.color("red")
    
    def curve():
        for i in range(200):
            turtle.right(1)
            turtle.forward(1)
            
    turtle.begin_fill()
    turtle.left(140)
    turtle.forward(111.65)
    curve()
    turtle.left(120)
    curve()
    turtle.forward(111.65)
    turtle.end_fill()
    
    turtle.hideturtle()
    turtle.done()

if __name__ == "__main__":
    heart()