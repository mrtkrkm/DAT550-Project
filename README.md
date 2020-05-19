# Data Mining DAT550-Project
# Building a chatbot about COVID-19 with Task Modelling
Murat korkmaz

In order to talk with chatbot use one of the StartConversations jupyter file.
```python
MakeConv=MakeConv(Embedding_Path, Task_Path, device, name, meta)
```

Change the embedding_path with your embedding path\
Also change the meta path with your metadata.csv path

If you want to use StartConversationsLSTM jupyter file

first you have to change the code below in Encoder.py
```python
self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
```
with
```python
self.gru = nn.LSTM(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
```

Also change the code below in Decoder.py
```python
self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
```
with
```python 
self.gru = nn.LSTM(hidden_size, hidden_size, n_layers*2, dropout=(0 if n_layers == 1 else dropout))
```

Also 

in the MakeConv.py file
```python
pathf = 'C:/Users/mkork/Desktop/dat'
```
Change the path with your path, which contains the trained weights.