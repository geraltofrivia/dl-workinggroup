#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[6]:


torch.manual_seed(42)


# In[7]:


# Only one data instance
x = torch.tensor(10)
y = torch.tensor(2)


# In[8]:


# Simplest possible model?
m = torch.randn(1, requires_grad=True)
c = torch.randn(1, requires_grad=True)


print(m, c)


# In[9]:


fx = lambda x : (m*x) + c


# In[10]:


def mse(y_pred, y_true):
    return 0.5*(y_pred-y_true)**2


# In[11]:


lr = 0.01


# In[12]:


for i in range(2000):
    # Calcualte model predictions
    y_pred = fx(x)
    
    # Compare the prediction with our goal
    loss = mse(y_pred, y)
    print(f"Loss: {loss}\nTrue: {y}\nPred: {y_pred.item()}")
    
    # Reset the gradients before computing new ones
    if m.grad:
        m.grad.zero_()
        c.grad.zero_()
        
    # Compute new gradients: BACKPROPAGATE
    loss.backward()
    
    print(f"Parameters before update:\n\tm: {m.item()}\tgrad: {m.grad.item()}\n\tc: {c.item()}\tgrad: {c.grad.item()}")
#     print(m, m.grad, c, c.grad)
    with torch.no_grad():
        m.copy_(m - (lr*m.grad))
        c.copy_(m - (lr*m.grad))
    print(f"Parametrs after update:\n\tm: {m.item()}\tgrad: {m.grad.item()  if c.grad else None}\n\tc: {c.item()}\tgrad: {c.grad.item() if c.grad else None}")
#     print(m, m.grad, c, c.grad)
#     m.grad.zero_()
#     c.grad.zero_()
    print('------', i, '------')
    cmd = input().strip()
    if cmd in ['q', 'exit', 'break']:
        break


# In[ ]:




