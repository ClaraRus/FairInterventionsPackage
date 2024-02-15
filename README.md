# A Study of Pre-processing Fairness Intervention Methods for Ranking People
## Requirements
The use of CIF-Rank requires [R]([https://pages.github.com/](https://cran.r-project.org/bin/linux/ubuntu/fullREADME.html#installing-r)) to be installed.

## Install
```
git clone https://github.com/ClaraRus/FairInterventionsPackage.git
conda create -n fair_interventions python=3.8
conda activate fair_interventions
cd FairInterventions
pip install .
```
## Example Use

### CIF-Rank
```
model = CIFRank(query_col='Occupation', IV=['Gender', 'Nationality', 'Occupation'],
                MED=['Education', 'Experience', 'Languages'], DV='Score',
                control='female_non-european_surgeon', out_path=out_path, pos_th=0, run=0)

# generate fair data for the candidates
df_train_count, df_test_count = model.generate_fair_data(df_train, df_test)

# rank the candidates by the debiased score
df_train_reranked = df_train_count.groupby('Occupation').apply(lambda x: x.sort_values(by='Score_fair', ascending=False)).reset_index(drop=True)
df_test_reranked = df_test_count.groupby('Occupation').apply(lambda x: x.sort_values(by='Score_fair', ascending=False)).reset_index(drop=True)
```
### LFR
```
# init pre-processing fairness intervention
model = LearningFairRepresentations(query_col='Occupation', sensitive_col='Gender', score_col='Score', features_col=['Education', 'Experience', 'Languages'],
                                    k = 2, Ax=1, Ay=1, Az=1, out_path=out_path, pos_th=0, run=0, model_occ=True)
# generate fair data for the candidates
df_train_count, df_test_count = model.generate_fair_data(df_train, df_test)

# rank the candidates by the debiased score
df_train_reranked = df_train_count.groupby('Occupation').apply(lambda x: x.sort_values(by='Score_fair', ascending=False)).reset_index(drop=True)
df_test_reranked = df_test_count.groupby('Occupation').apply(lambda x: x.sort_values(by='Score_fair', ascending=False)).reset_index(drop=True)
```

### iFair
```
# init pre-processing fairness intervention
model = iFairRanking(query_col='Occupation', sensitive_col='Gender', score_col='Score', features_col=['Education', 'Experience', 'Languages'],
                                    k = 2, Ax=1, Az=1, out_path=out_path, pos_th=0, run=0, max_iter=100, nb_restarts=3, batch_size=1000, model_occ=True)

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
Ke Yang, Joshua R. Loftus, and Julia Stoyanovich. 2021. Causal intersectionality and fair ranking. In Symposium on Foundations of Responsible
Computing (FORC). 

[3]
Preethi Lahoti, Krishna P Gummadi, and Gerhard Weikum. 2019. ifair: Learning individually fair data representations for algorithmic decision
making. In 2019 IEEE 35th International Conference on Data Engineering (ICDE). IEEE, 1334–1345.

[4]
Zemel, Rich, et al. "Learning fair representations." International conference on machine learning. PMLR, 2013.
