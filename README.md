# DAT550-Project
Murat korkmaz

In order to talk with chatbot use one of the StartConversations jupyter file.
```python
MakeConv=MakeConv(Embedding_Path, Task_Path, device, name, meta)
```

Change the embedding_path with your path
Also change the meta path with your metadata.csv path

If you want to use StartConversationsGRU jupyter file

first you have to change the code below
```python
self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
'''