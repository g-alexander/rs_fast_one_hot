# Fast One Hot Encoder

This is simple package for fast one hot encoding, implemented in Rust.

## Installation
```bash
pip install rs_fast_one_hot --upgrade
```

## Usage
```python
from rs_fast_one_hot import OneHotTransformer

data = ['1a', '1a', '2h', '5j', '8n', '8n', '5j']

transformer = OneHotTransformer()
transformer.fit(data)
res = transformer.transform(data)

print("Encoded data:", res.toarray())
```

for multithread transform:
```python
transformer = OneHotTransformer(n_jobs=5)
```

to back to single thread:
```python
transformer.to_single_thread()
```
complete example
```pycon
from rs_fast_one_hot import OneHotTransformer

data = ['1a', '1a', '2h', '5j', '8n', '8n', '5j']

transformer = OneHotTransformer(5)
transformer.fit(data)

res = transformer.transform(data)
print("Encoded data before save", res.toarray())

transformer.to_single_thread() # convert to single thread for prom usage
with open('transformer.pkl', 'wb') as f:
    pickle.dump(transformer, f)

# Load transformer from disk
with open('transformer.pkl', 'rb') as f:
    transformer2 = pickle.load(f)
res2 = transformer2.transform(data)
print("Encoded data after load from disk", res2.toarray())
```