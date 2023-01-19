import torch
LENGTH = 1.0
TOTAL_TIME = 1.0    
    
    
    
    
    
    
x_domain = [0.0, LENGTH]; n_points_x = 5
t_domain = [0.0, TOTAL_TIME]; n_points_t = 5
    

x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points_x, requires_grad=True)
t_raw = torch.linspace(t_domain[0], t_domain[1], steps=n_points_t, requires_grad=True)
grids = torch.meshgrid(x_raw, t_raw, indexing="ij")
    
print("x_raw: ", x_raw)
print("t_raw: ", t_raw)
print("shape of grids: ", type(grids))
print("grids: ", grids)
print("shape of first tuple: ", grids[0].shape)
print("grids first component:", grids[0])

print("shape of second tuple:", grids[1].shape)
print("grids second component:", grids[1])

print("grids i j component:", grids[0][4,0])


x = grids[0].flatten().reshape(-1, 1)
y = grids[1].flatten().reshape(-1, 1)

print("x:", x)
print("y:", y)