【WSDM-2019 SimGNN】[SimGNN: A Neural Network Approach to Fast Graph Similarity Computation](https://arxiv.org/abs/1808.05689)
![image](https://github.com/XiaShan1227/SimGNN/assets/67092235/add4d865-e696-4708-a261-04cb172bcf71)

```python
python main.py --exp_name=SimGNN
```

| Parameter | Value      |
|:--------:| -------------:|
| left-aligned Batch size | left-aligned 16 |
| left-aligned Bins  | left-aligned 16 |
| left-aligned Bottle neck neurons | left-aligned 16 |
| left-aligned Dropout | left-aligned 0.5 |
| left-aligned Batch size | left-aligned 16 |

  --filters-1             INT         Number of filter in 1st GCN layer.       Default is 128.
  --filters-2             INT         Number of filter in 2nd GCN layer.       Default is 64. 
  --filters-3             INT         Number of filter in 3rd GCN layer.       Default is 32.
  --tensor-neurons        INT         Neurons in tensor network layer.         Default is 16.
  --bottle-neck-neurons   INT         Bottle neck layer neurons.               Default is 16.
  --bins                  INT         Number of histogram bins.                Default is 16.
  --batch-size            INT         Number of pairs processed per batch.     Default is 128. 
  --epochs                INT         Number of SimGNN training epochs.        Default is 5.
  --dropout               FLOAT       Dropout rate.                            Default is 0.5.
  --learning-rate         FLOAT       Learning rate.                           Default is 0.001.
  --weight-decay          FLOAT       Weight decay.                            Default is 10^-5.
  --histogram             BOOL        Include histogram features.              Default is False.




<img src="https://github.com/XiaShan1227/SimGNN/assets/67092235/782867c0-62bb-43cd-92b7-abba4f3f3b2a" alt="Image" width="600" height="500">


Code Framework Reference: [SimGNN](https://github.com/benedekrozemberczki/SimGNN)
