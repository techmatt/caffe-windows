rem compute_image_mean.exe --backend=leveldb circles-train-leveldb circles-mean.binaryproto
rem compute_image_mean.exe --backend=leveldb simulation-train-leveldb simulation-mean.binaryproto
rem compute_image_mean.exe --backend=leveldb clouds-train-leveldb clouds-mean.binaryproto
compute_image_mean.exe --backend=leveldb particles-train-leveldb particles-mean.binaryproto
pause