## Feature Providers

Abstractions for methods of pulling features. The two main ones are:

* `sidecar`: Locate the pre-computed feature from a file path
* `online`: Generally ill-advised for large jobs, but compute the features on-the-fly during training or inference. Great for small-batch and proof-of-concept.