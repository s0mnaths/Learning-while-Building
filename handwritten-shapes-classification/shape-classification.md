# Hand-Written shapes classification using DeepLearning (PyTorch)

#### Importing libraries

`pandas` for playing with .csv

`numpy` for dealing with images

`torch.nn` for defining neural network

`torch.nn.functional`

`torch.optim`

`torchvision.transforms` for defining transforms

`torch.utils.data.Dataloader` for creating dataloader

`matplotlib.pyplot` for plotting images/graphs



```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split 
import matplotlib.pyplot as plt
import random
```


```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
```

    Using cpu device


## Load dataset
Loading the dataset for training+validating and for testing
then splitting them into images and labels


```python
df_fullDataset = pd.read_csv('train_data.csv')
df_fullDataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>P1</th>
      <th>P2</th>
      <th>P3</th>
      <th>P4</th>
      <th>P5</th>
      <th>P6</th>
      <th>P7</th>
      <th>P8</th>
      <th>P9</th>
      <th>...</th>
      <th>P775</th>
      <th>P776</th>
      <th>P777</th>
      <th>P778</th>
      <th>P779</th>
      <th>P780</th>
      <th>P781</th>
      <th>P782</th>
      <th>P783</th>
      <th>P784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>...</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>254</td>
      <td>255</td>
      <td>255</td>
      <td>254</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>0</td>
      <td>...</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>255</td>
      <td>254</td>
      <td>255</td>
      <td>255</td>
      <td>254</td>
      <td>255</td>
      <td>253</td>
      <td>255</td>
      <td>254</td>
      <td>...</td>
      <td>254</td>
      <td>255</td>
      <td>254</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>...</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>253</td>
      <td>...</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>1</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>...</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
    </tr>
    <tr>
      <th>196</th>
      <td>1</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>...</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
    </tr>
    <tr>
      <th>197</th>
      <td>1</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>254</td>
      <td>...</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
    </tr>
    <tr>
      <th>198</th>
      <td>1</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>...</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
    </tr>
    <tr>
      <th>199</th>
      <td>1</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>...</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 785 columns</p>
</div>



### Get the image pixel values and labels


```python
fullDataset_images = df_fullDataset.iloc[:, 1:]
fullDataset_labels = df_fullDataset.iloc[:, 0]
```

### Transforms
Defining image transforms. Using `Compose` we chain different transforms.

Here we transform image csv data to PIL image, then to Tensor, and then they are normalized. 

Tensor images with a float dtype are expected to have values in `[0,1)`.

`Normalize` does the following for each channel:

`image = (image - mean) / std`
The parameters for Normalize are `(mean, std)`

The parameters mean, std are passed as 0.5, 0.5. This will normalize the image in the range `[-1,1]`. For example, the minimum value 0 will be converted to (0-0.5)/0.5=-1, the maximum value of 1 will be converted to (1-0.5)/0.5=1.


```python
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))
])
```

### Custom Dataset
Defining custom dataset. 
We create a class with three functions. `__init__` initializes, `__len__` is returns the length of dataset, `__getitem__` returns specific data.


```python
class ShapesDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X.iloc[i, :]
        data = np.asarray(data).astype(np.uint8).reshape(28, 28, 1)
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data
```


```python
full_data = ShapesDataset(fullDataset_images, fullDataset_labels, transform)
```

### Plotting the Dataset


```python
labels_map = {
    0: "Circle",
    1: "Triangle",
    2: "Square",
}
figure = plt.figure(figsize=(4, 4))
cols, rows = 4, 4
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(full_data), size=(1,)).item()
    img, label = full_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```


    
![png](shape-classification_files/shape-classification_13_0.png)
    


### Splitting the Full Dataset
Splitting using the `random_split` function(takes in the full ds and sizes) from torch.utils.data


```python
train_size = int(0.7*len(full_data))
val_size = len(full_data) - train_size

train_data, val_data = random_split(full_data,[train_size,val_size])
```

### Dataloader
While training a model, we typically want to pass samples in _“minibatches”_, reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing to speed up data retrieval.

Dataloader is an iterable that does this


```python
trainloader = DataLoader(train_data, batch_size=10, shuffle=True)
valloader = DataLoader(val_data, batch_size=10, shuffle=True)
```

## Building and Training the Model
### Defining the NN
We initialize the NN in the `__init__` method, and operations are implemented on the input data in the `forward` method.

`nn.flatten` converts each 2D 28x28 imaage into continous array of 784 pixel values.

`nn.Linear` applies a linear transformation on the input using its stored weights and biases.

Non-linear activations are what create the complex mappings between the model’s inputs and outputs. They are applied after linear transformations to introduce nonlinearity, helping neural networks learn a wide variety of phenomena. There are many functions/activations, but here we use `nnReLU`

`nn.Sequential` is an ordered container of modules/fns. 


```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

We create an instance of `NeuralNetwork` and move it to device.


```python
model = NeuralNetwork().to(device)
print(model)
```

    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=3, bias=True)
      )
    )


## Optimization Loop
We train and optimize our model using the Optimization loop, whose each iteration is called an **epoch**. Each epoch has two main parts:
1. **Train Loop** - Iterate over training dataset
2. **Validation Loop** - Iterate over validation dataset to check if model performance is improving.

### Loss Function
Loss function measures the degree of dissimilarity of obtained result to the target value, and we want to minimize this during training.
To calculate the loss we make a prediction using the inputs of our given date and compare it against the true data label value.

Here we use `nn.CrossEntropyLoss` (combination of `nn.LogSoftmax` and `nn.NLLLoss`).


```python
loss_fn = nn.CrossEntropyLoss()
```

### Optimizer
Optimzation is the process of adjusting model parameters in each training step, to reduce model error. 

Different Optimization algorithms define how this process is performed. Here we use `SGD` optimizer, i.e. **Stochastic Gradient Descent**. We also pass in the *Learning rate* as the second parameter.


```python
optimizer = optim.SGD(model.parameters(), lr=1e-3)
```


```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()  #resets the gradient of model parameter, otherwise they'll add up in each iteration
        loss.backward()   #PyTorch deposits the gradients of the loss w.r.t. each parameter
        optimizer.step()   #adjusts the parameters by the gradients collected in the backward pass

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```


```python
def val(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```


```python
epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(trainloader, model, loss_fn, optimizer)
    val(valloader, model, loss_fn)
print("Done!")
```

    Epoch 1
    -------------------------------
    loss: 1.106827  [    0/  140]
    Test Error: 
     Accuracy: 56.7%, Avg loss: 1.084811 
    
    Epoch 2
    -------------------------------
    loss: 1.061358  [    0/  140]
    Test Error: 
     Accuracy: 56.7%, Avg loss: 1.067237 
    
    Epoch 3
    -------------------------------
    loss: 1.048741  [    0/  140]
    Test Error: 
     Accuracy: 56.7%, Avg loss: 1.050747 
    
    Epoch 4
    -------------------------------
    loss: 1.062198  [    0/  140]
    Test Error: 
     Accuracy: 56.7%, Avg loss: 1.035249 
    
    Epoch 5
    -------------------------------
    loss: 1.010120  [    0/  140]
    Test Error: 
     Accuracy: 56.7%, Avg loss: 1.020594 
    
    Epoch 6
    -------------------------------
    loss: 1.019774  [    0/  140]
    Test Error: 
     Accuracy: 56.7%, Avg loss: 1.006494 
    
    Epoch 7
    -------------------------------
    loss: 0.965433  [    0/  140]
    Test Error: 
     Accuracy: 56.7%, Avg loss: 0.992823 
    
    Epoch 8
    -------------------------------
    loss: 0.975169  [    0/  140]
    Test Error: 
     Accuracy: 63.3%, Avg loss: 0.979320 
    
    Epoch 9
    -------------------------------
    loss: 0.982147  [    0/  140]
    Test Error: 
     Accuracy: 63.3%, Avg loss: 0.965874 
    
    Epoch 10
    -------------------------------
    loss: 0.877847  [    0/  140]
    Test Error: 
     Accuracy: 63.3%, Avg loss: 0.952622 
    
    Epoch 11
    -------------------------------
    loss: 0.886880  [    0/  140]
    Test Error: 
     Accuracy: 63.3%, Avg loss: 0.939572 
    
    Epoch 12
    -------------------------------
    loss: 0.889644  [    0/  140]
    Test Error: 
     Accuracy: 63.3%, Avg loss: 0.926426 
    
    Epoch 13
    -------------------------------
    loss: 0.945943  [    0/  140]
    Test Error: 
     Accuracy: 63.3%, Avg loss: 0.913177 
    
    Epoch 14
    -------------------------------
    loss: 0.794504  [    0/  140]
    Test Error: 
     Accuracy: 63.3%, Avg loss: 0.899988 
    
    Epoch 15
    -------------------------------
    loss: 0.811973  [    0/  140]
    Test Error: 
     Accuracy: 63.3%, Avg loss: 0.886862 
    
    Epoch 16
    -------------------------------
    loss: 0.998779  [    0/  140]
    Test Error: 
     Accuracy: 63.3%, Avg loss: 0.873778 
    
    Epoch 17
    -------------------------------
    loss: 0.877213  [    0/  140]
    Test Error: 
     Accuracy: 63.3%, Avg loss: 0.860638 
    
    Epoch 18
    -------------------------------
    loss: 0.838639  [    0/  140]
    Test Error: 
     Accuracy: 63.3%, Avg loss: 0.847694 
    
    Epoch 19
    -------------------------------
    loss: 0.761388  [    0/  140]
    Test Error: 
     Accuracy: 68.3%, Avg loss: 0.834590 
    
    Epoch 20
    -------------------------------
    loss: 0.917019  [    0/  140]
    Test Error: 
     Accuracy: 68.3%, Avg loss: 0.821574 
    
    Epoch 21
    -------------------------------
    loss: 0.869925  [    0/  140]
    Test Error: 
     Accuracy: 68.3%, Avg loss: 0.808563 
    
    Epoch 22
    -------------------------------
    loss: 0.761903  [    0/  140]
    Test Error: 
     Accuracy: 68.3%, Avg loss: 0.795796 
    
    Epoch 23
    -------------------------------
    loss: 0.659732  [    0/  140]
    Test Error: 
     Accuracy: 68.3%, Avg loss: 0.783049 
    
    Epoch 24
    -------------------------------
    loss: 0.765023  [    0/  140]
    Test Error: 
     Accuracy: 70.0%, Avg loss: 0.770356 
    
    Epoch 25
    -------------------------------
    loss: 0.753673  [    0/  140]
    Test Error: 
     Accuracy: 70.0%, Avg loss: 0.757907 
    
    Epoch 26
    -------------------------------
    loss: 0.584563  [    0/  140]
    Test Error: 
     Accuracy: 70.0%, Avg loss: 0.745238 
    
    Epoch 27
    -------------------------------
    loss: 0.707059  [    0/  140]
    Test Error: 
     Accuracy: 80.0%, Avg loss: 0.732895 
    
    Epoch 28
    -------------------------------
    loss: 0.642416  [    0/  140]
    Test Error: 
     Accuracy: 80.0%, Avg loss: 0.720618 
    
    Epoch 29
    -------------------------------
    loss: 0.801895  [    0/  140]
    Test Error: 
     Accuracy: 80.0%, Avg loss: 0.708429 
    
    Epoch 30
    -------------------------------
    loss: 0.607397  [    0/  140]
    Test Error: 
     Accuracy: 80.0%, Avg loss: 0.696509 
    
    Epoch 31
    -------------------------------
    loss: 0.652709  [    0/  140]
    Test Error: 
     Accuracy: 80.0%, Avg loss: 0.684499 
    
    Epoch 32
    -------------------------------
    loss: 0.734066  [    0/  140]
    Test Error: 
     Accuracy: 80.0%, Avg loss: 0.672773 
    
    Epoch 33
    -------------------------------
    loss: 0.686141  [    0/  140]
    Test Error: 
     Accuracy: 83.3%, Avg loss: 0.661347 
    
    Epoch 34
    -------------------------------
    loss: 0.625041  [    0/  140]
    Test Error: 
     Accuracy: 83.3%, Avg loss: 0.649877 
    
    Epoch 35
    -------------------------------
    loss: 0.579901  [    0/  140]
    Test Error: 
     Accuracy: 83.3%, Avg loss: 0.638937 
    
    Epoch 36
    -------------------------------
    loss: 0.499554  [    0/  140]
    Test Error: 
     Accuracy: 83.3%, Avg loss: 0.627979 
    
    Epoch 37
    -------------------------------
    loss: 0.518245  [    0/  140]
    Test Error: 
     Accuracy: 83.3%, Avg loss: 0.617334 
    
    Epoch 38
    -------------------------------
    loss: 0.407886  [    0/  140]
    Test Error: 
     Accuracy: 83.3%, Avg loss: 0.606820 
    
    Epoch 39
    -------------------------------
    loss: 0.467602  [    0/  140]
    Test Error: 
     Accuracy: 83.3%, Avg loss: 0.596545 
    
    Epoch 40
    -------------------------------
    loss: 0.566936  [    0/  140]
    Test Error: 
     Accuracy: 83.3%, Avg loss: 0.586572 
    
    Epoch 41
    -------------------------------
    loss: 0.386981  [    0/  140]
    Test Error: 
     Accuracy: 83.3%, Avg loss: 0.576630 
    
    Epoch 42
    -------------------------------
    loss: 0.571752  [    0/  140]
    Test Error: 
     Accuracy: 83.3%, Avg loss: 0.566959 
    
    Epoch 43
    -------------------------------
    loss: 0.602507  [    0/  140]
    Test Error: 
     Accuracy: 83.3%, Avg loss: 0.557570 
    
    Epoch 44
    -------------------------------
    loss: 0.532299  [    0/  140]
    Test Error: 
     Accuracy: 83.3%, Avg loss: 0.547922 
    
    Epoch 45
    -------------------------------
    loss: 0.565234  [    0/  140]
    Test Error: 
     Accuracy: 83.3%, Avg loss: 0.538717 
    
    Epoch 46
    -------------------------------
    loss: 0.425777  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.529639 
    
    Epoch 47
    -------------------------------
    loss: 0.431055  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.520863 
    
    Epoch 48
    -------------------------------
    loss: 0.381543  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.512188 
    
    Epoch 49
    -------------------------------
    loss: 0.519722  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.503616 
    
    Epoch 50
    -------------------------------
    loss: 0.361820  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.495467 
    
    Epoch 51
    -------------------------------
    loss: 0.404488  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.487345 
    
    Epoch 52
    -------------------------------
    loss: 0.481953  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.479366 
    
    Epoch 53
    -------------------------------
    loss: 0.361616  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.471473 
    
    Epoch 54
    -------------------------------
    loss: 0.280828  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.463821 
    
    Epoch 55
    -------------------------------
    loss: 0.401033  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.455804 
    
    Epoch 56
    -------------------------------
    loss: 0.315308  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.448352 
    
    Epoch 57
    -------------------------------
    loss: 0.421219  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.441245 
    
    Epoch 58
    -------------------------------
    loss: 0.360474  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.434118 
    
    Epoch 59
    -------------------------------
    loss: 0.293255  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.427021 
    
    Epoch 60
    -------------------------------
    loss: 0.193565  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.420586 
    
    Epoch 61
    -------------------------------
    loss: 0.382146  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.413731 
    
    Epoch 62
    -------------------------------
    loss: 0.187756  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.407377 
    
    Epoch 63
    -------------------------------
    loss: 0.313646  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.400940 
    
    Epoch 64
    -------------------------------
    loss: 0.284936  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.394505 
    
    Epoch 65
    -------------------------------
    loss: 0.337026  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.388367 
    
    Epoch 66
    -------------------------------
    loss: 0.212516  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.382117 
    
    Epoch 67
    -------------------------------
    loss: 0.431135  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.376309 
    
    Epoch 68
    -------------------------------
    loss: 0.413738  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.370151 
    
    Epoch 69
    -------------------------------
    loss: 0.356926  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.364629 
    
    Epoch 70
    -------------------------------
    loss: 0.272933  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.359268 
    
    Epoch 71
    -------------------------------
    loss: 0.285254  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.353998 
    
    Epoch 72
    -------------------------------
    loss: 0.198413  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.348741 
    
    Epoch 73
    -------------------------------
    loss: 0.289075  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.343364 
    
    Epoch 74
    -------------------------------
    loss: 0.295409  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.338618 
    
    Epoch 75
    -------------------------------
    loss: 0.275455  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.333677 
    
    Epoch 76
    -------------------------------
    loss: 0.130791  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.328522 
    
    Epoch 77
    -------------------------------
    loss: 0.152062  [    0/  140]
    Test Error: 
     Accuracy: 88.3%, Avg loss: 0.323690 
    
    Epoch 78
    -------------------------------
    loss: 0.122701  [    0/  140]
    Test Error: 
     Accuracy: 91.7%, Avg loss: 0.318982 
    
    Epoch 79
    -------------------------------
    loss: 0.230123  [    0/  140]
    Test Error: 
     Accuracy: 91.7%, Avg loss: 0.314492 
    
    Epoch 80
    -------------------------------
    loss: 0.183704  [    0/  140]
    Test Error: 
     Accuracy: 91.7%, Avg loss: 0.309632 
    
    Epoch 81
    -------------------------------
    loss: 0.324070  [    0/  140]
    Test Error: 
     Accuracy: 91.7%, Avg loss: 0.305106 
    
    Epoch 82
    -------------------------------
    loss: 0.230650  [    0/  140]
    Test Error: 
     Accuracy: 91.7%, Avg loss: 0.300969 
    
    Epoch 83
    -------------------------------
    loss: 0.348526  [    0/  140]
    Test Error: 
     Accuracy: 91.7%, Avg loss: 0.296850 
    
    Epoch 84
    -------------------------------
    loss: 0.151113  [    0/  140]
    Test Error: 
     Accuracy: 91.7%, Avg loss: 0.292836 
    
    Epoch 85
    -------------------------------
    loss: 0.181945  [    0/  140]
    Test Error: 
     Accuracy: 91.7%, Avg loss: 0.288802 
    
    Epoch 86
    -------------------------------
    loss: 0.208451  [    0/  140]
    Test Error: 
     Accuracy: 93.3%, Avg loss: 0.284652 
    
    Epoch 87
    -------------------------------
    loss: 0.144053  [    0/  140]
    Test Error: 
     Accuracy: 93.3%, Avg loss: 0.280940 
    
    Epoch 88
    -------------------------------
    loss: 0.201365  [    0/  140]
    Test Error: 
     Accuracy: 93.3%, Avg loss: 0.277198 
    
    Epoch 89
    -------------------------------
    loss: 0.287982  [    0/  140]
    Test Error: 
     Accuracy: 93.3%, Avg loss: 0.272871 
    
    Epoch 90
    -------------------------------
    loss: 0.139104  [    0/  140]
    Test Error: 
     Accuracy: 93.3%, Avg loss: 0.269834 
    
    Epoch 91
    -------------------------------
    loss: 0.105491  [    0/  140]
    Test Error: 
     Accuracy: 93.3%, Avg loss: 0.265465 
    
    Epoch 92
    -------------------------------
    loss: 0.202622  [    0/  140]
    Test Error: 
     Accuracy: 93.3%, Avg loss: 0.262262 
    
    Epoch 93
    -------------------------------
    loss: 0.192476  [    0/  140]
    Test Error: 
     Accuracy: 93.3%, Avg loss: 0.258925 
    
    Epoch 94
    -------------------------------
    loss: 0.115338  [    0/  140]
    Test Error: 
     Accuracy: 93.3%, Avg loss: 0.255425 
    
    Epoch 95
    -------------------------------
    loss: 0.121925  [    0/  140]
    Test Error: 
     Accuracy: 93.3%, Avg loss: 0.251914 
    
    Epoch 96
    -------------------------------
    loss: 0.124791  [    0/  140]
    Test Error: 
     Accuracy: 98.3%, Avg loss: 0.248398 
    
    Epoch 97
    -------------------------------
    loss: 0.299985  [    0/  140]
    Test Error: 
     Accuracy: 98.3%, Avg loss: 0.245184 
    
    Epoch 98
    -------------------------------
    loss: 0.232400  [    0/  140]
    Test Error: 
     Accuracy: 98.3%, Avg loss: 0.242398 
    
    Epoch 99
    -------------------------------
    loss: 0.226371  [    0/  140]
    Test Error: 
     Accuracy: 98.3%, Avg loss: 0.239478 
    
    Epoch 100
    -------------------------------
    loss: 0.208218  [    0/  140]
    Test Error: 
     Accuracy: 98.3%, Avg loss: 0.235953 
    
    Done!


### Saving the model


```python
torch.save(model.state_dict(), "./model.pth")
print("Saved PyTorch Model State to model.pth")
```

    Saved PyTorch Model State to model.pth


## Testing the Model
### Loading the model


```python
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
```




    <All keys matched successfully>




```python
classes = [
    "Circle",
    "Triangle",
    "Square",
]

model.eval()
```




    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=3, bias=True)
      )
    )




```python
for i in range(10):
    rand_idx = random.randint(0,len(val_data)-1)
    x, y = val_data[rand_idx][0], val_data[rand_idx][1]

    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

    Predicted: "Triangle", Actual: "Circle"
    Predicted: "Square", Actual: "Square"
    Predicted: "Circle", Actual: "Circle"
    Predicted: "Triangle", Actual: "Triangle"
    Predicted: "Circle", Actual: "Circle"
    Predicted: "Triangle", Actual: "Triangle"
    Predicted: "Triangle", Actual: "Triangle"
    Predicted: "Square", Actual: "Square"
    Predicted: "Square", Actual: "Square"
    Predicted: "Triangle", Actual: "Triangle"

