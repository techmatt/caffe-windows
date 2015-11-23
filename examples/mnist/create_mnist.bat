
rd /s /q mnist-train-leveldb
rd /s /q mnist-test-leveldb
convert_mnist_data.exe data/train-images-idx3-ubyte data/train-labels-idx1-ubyte mnist-train-leveldb --backend=leveldb
convert_mnist_data.exe data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte mnist-test-leveldb --backend=leveldb

rd /s /q mnist-train-lmdb
rd /s /q mnist-test-lmdb
convert_mnist_data.exe data/train-images-idx3-ubyte data/train-labels-idx1-ubyte mnist-train-lmdb --backend=lmdb
convert_mnist_data.exe data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte mnist-test-lmdb --backend=lmdb

pause