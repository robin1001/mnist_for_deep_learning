# Experiment on Mnist

Some deep learning experiments on Mnist dataset, for quick implementation and verification.

## SVD Compression

> Restructuring of Deep Neural Network Acoustic Models with Singular Value Decomposition 

| Model(hidden layer) | Number of parameters | Compression ratio | accuracy | SVD retraining accuracy |
|---------------------|----------------------|-------------------|----------|-------------------------|
| DNN baseline(512)   | 932362               | 1.0               | 0.9836   | -                       |
| SVD(256)            | 862730               | 0.925317          | 0.9800   | 0.9730                  |
| SVD(128)            | 434698               | 0.466233          | 0.9800   | 0.9687                  |
| SVD(64)             | 220682               | 0.236691          | 0.9791   | 0.9806                  |
| SVD(32)             | 113674               | 0.121920          | 0.9606   | 0.9782                  |
| SVD(16)             | 60170                | 0.064535          | 0.8141   | 0.9744                  |
| SVD(8)              | 33418                | 0.035842          | 0.5299   | 0.9420                  |

* Some degradation on SVD retraining when, maybe need more tuning work.
* I think SVD retraining still helps, epecially when compress hard.

