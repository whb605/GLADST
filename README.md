# GLADST
This is the code for paper "Discriminative Graph Level Anomaly Detection via  Dual-students-teacher Model".


## Data Preparation

 You may download the datasets from the urls listed in the paper. For the datasets from Tox21, you should download both Tox21_xxx_training and Tox21_xxx_testing.

## Requirements

Requirements are listed in the requirements.txt

## Train

For datasets from Tox21, run the following code.

	python main.py --DS Tox21_HSE_training --feature deg-num
	
For the rest datasets, run the following code. For datasets with node attributes, feature chooses default, otherwise deg-num.

	python main.py --DS DHFR --feature default



## Citation
```bibtex
@inproceedings{
}
```
