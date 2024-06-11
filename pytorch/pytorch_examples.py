'''
Start the course 
https://www.youtube.com/watch?v=c36lUUr864M

make examples and change your code

'''

import torch

# # Scalar
# t1 = torch.tensor(4.)
# print(t1)
# print(t1.dtype)
# print(t1.shape)

# # # Vector 
# t2 = torch.tensor([1., 2, 3, 4])
# print(t2)
# print(t2.dtype)
# print(t2.shape)

# # Matrix
# t3 = torch.tensor([[1., 2, 3, 4], 
#                    [5, 2, 3, 4], 
#                    [6, 2, 3, 4], 
#                    [7, 2, 3, 4]], dtype=torch.float64)
# print(t3)
# print(t3.dtype)
# print(t3.shape)

# # Tensor
# t4 = torch.tensor([[[[1., 2],  [3, 4]], [[5, 6], [7, 8]]], [[[1., 2],  [3, 4]], [[5, 6], [7, 8]]]])
# print(t4)
# print(t4.dtype)
# print(t4.shape)

# Tensor operations and gradients
x = torch.tensor(3.)
w = torch.tensor(2., requires_grad = True)
b = torch.tensor(10., requires_grad = True)
print(x, w, b)

y = w * x + b * torch.sqrt(b)
print(y) 

# Compute derivates
y.backward()

# Print gradients
print('dy/dx ', x.grad)
print('dy/dw ', w.grad)
print('dy/db ', b.grad)


