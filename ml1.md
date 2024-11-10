def function(x):
    return (x + 3)**2
def derivative(x):
    return 2 * (x + 3)
  
x = 2
alpha = 0.1
iterations = 1000
tolerance = 1e-6
for i in range(iterations):
    gradient = derivative(x)
    new_x = x - alpha * gradient
    if abs(new_x - x) < tolerance:
        break
    x = new_x
print("Local minima at x =", x)
print("Minimum value of the function y =", function(x))
