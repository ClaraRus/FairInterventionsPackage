# Preprocessing Fairness Intervention Package
## Requirements
The use of CIF-Rank requires [R](https://cran.r-project.org/bin/linux/ubuntu/fullREADME.html#installing-r) to be installed.

## Install
```
git clone https://github.com/ClaraRus/FairInterventionsPackage.git
conda create -n fair_interventions python=3.8
conda activate fair_interventions
cd FairInterventions
pip install .
```
## Example Use

### CIF-Rank [3]
```
# init pre-processing fairness intervention
# <out_path> - output path to save the data and the intermediary steps
# IV - predictors in the causal model. By integrating the 'Occupation' in addition to the sensitive informatioin (e.g. gender, natinality) the model will account for the varying bias directions encoded in each occupation.

model = CIFRank(query_col='Occupation', IV=['Gender', 'Nationality', 'Occupation'],
                MED=['Education', 'Experience', 'Languages'], DV='Score',
                control='female_non-european_surgeon', out_path=<out_path>)
```
### LFR [5]
```
# init pre-processing fairness intervention
# <out_path> - output path to save the data and the intermediary steps
# k - number of protoypes
# Ax - hyperparameter for Lx, the data loss that should optimize for keeping the new representations as close as possible to the original ones
# Ay - hyperparameter for Ly, the utility loss that should ensure that representations are still useful
# Az - hyperparameter for Lz, the group fairness loss

model = LearningFairRepresentations(query_col='Occupation', sensitive_col='Gender', score_col='Score', features_col=['Education', 'Experience', 'Languages'],
                                    k = 2, Ax=1, Ay=1, Az=1, out_path=<out_path>, model_occ=True)
```

### iFair [4]
```
# init pre-processing fairness intervention
# <out_path> - output path to save the data and the intermediary steps
# Ax - hyperparameter for Lx, the data loss that should optimize for keeping the new representations as close as possible to the original ones
# Az - hyperparameter for Lz, the group fairness loss

model = iFairRanking(query_col='Occupation', sensitive_col='Gender', score_col='Score', features_col=['Education', 'Experience', 'Languages'],
                                    k = 2, Ax=1, Az=1, out_path=<out_path>, model_occ=True)

```
### Generate fair data
```
# generate fair data for the candidates
df_train_count, df_test_count = model.generate_fair_data(df_train, df_test)

# rank the candidates by the debiased score
df_train_reranked = df_train_count.groupby('Occupation').apply(lambda x: x.sort_values(by='Score_fair', ascending=False)).reset_index(drop=True)
df_test_reranked = df_test_count.groupby('Occupation').apply(lambda x: x.sort_values(by='Score_fair', ascending=False)).reset_index(drop=True)
```

## References
[1]
Rus, Clara, Maarten de Rijke, and Andrew Yates. "A Study of Pre-processing Fairness Intervention
Methods for Ranking People." (2024).

[2]
Rus, Clara, Maarten de Rijke, and Andrew Yates. "Counterfactual Representations for Intersectional Fair Ranking in Recruitment." (2023).

[3]
Ke Yang, Joshua R. Loftus, and Julia Stoyanovich. 2021. Causal intersectionality and fair ranking. In Symposium on Foundations of Responsible
Computing (FORC). 

[4]
Preethi Lahoti, Krishna P Gummadi, and Gerhard Weikum. 2019. ifair: Learning individually fair data representations for algorithmic decision
making. In 2019 IEEE 35th International Conference on Data Engineering (ICDE). IEEE, 1334–1345.

[5]
Zemel, Rich, et al. "Learning fair representations." International conference on machine learning. PMLR, 2013.
