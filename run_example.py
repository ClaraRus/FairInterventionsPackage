import os
import numpy as np
import pandas as pd

from FairInterventions.fairness_method_iFair import iFairRanking
from FairInterventions.fairness_method_LFR import LearningFairRepresentations
from FairInterventions.fairness_method_CIFRank import CIFRank


def create_train_test_split(df):
    # Splitting the dataset into training and testing sets
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for occupation in df['Occupation'].unique():
        # Get indices for each occupation
        occupation_indices = df[df['Occupation'] == occupation].index.tolist()

        # Shuffle indices to ensure randomness
        np.random.shuffle(occupation_indices)

        # Calculate the number of samples for testing (30%)
        num_test_samples = int(0.3 * len(occupation_indices))

        # Separate testing and training indices
        test_indices = occupation_indices[:num_test_samples]
        train_indices = occupation_indices[num_test_samples:]

        # Append data to train and test dataframes
        df_train = pd.concat([df_train, df.loc[train_indices]])
        df_test = pd.concat([df_test, df.loc[test_indices]])

    # Reset index for both dataframes
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    return df_train, df_test


def create_dummy_data():
    # Define the number of candidates for each occupation
    num_nurses = 40
    num_surgeons = 40

    # Generate gender for nurses and surgeons
    nurse_genders = np.random.choice(['female', 'male'], size=num_nurses, p=[0.8, 0.2])
    surgeon_genders = np.random.choice(['female', 'male'], size=num_surgeons, p=[0.2, 0.8])

    # Generate nationality for all candidates
    nationalities = np.random.choice(['european', 'non-european'], size=num_nurses + num_surgeons)

    # Generate other columns
    occupations = ['nurse'] * num_nurses + ['surgeon'] * num_surgeons
    genders = np.concatenate([nurse_genders, surgeon_genders])
    educations = np.random.rand(num_nurses + num_surgeons)
    experiences = np.random.rand(num_nurses + num_surgeons)
    languages = np.random.rand(num_nurses + num_surgeons)
    scores = np.random.rand(num_nurses + num_surgeons)

    # Create the dataframe
    data = {
        'Occupation': occupations,
        'Gender': genders,
        'Nationality': nationalities,
        'Education': educations,
        'Experience': experiences,
        'Languages': languages,
        'Score': scores
    }

    df = pd.DataFrame(data)
    return df

# Create dummy data
# Score - represent the relevance of the candidate for the given occupation (this is used to rank the candidates)
# Occupation - name of the occupation the candidates applied to
# Gender, Nationality - sensitive information of the candidates
# Education, Experience, Languages - features of the candidates
df_train, df_test = create_train_test_split(create_dummy_data())


################################ Example CIFRank ################################
# Output path to save the data and the intermediary steps
out_path = './FairInterventions/out/CIFRank'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# init pre-processing fairness intervention
model = CIFRank(query_col='Occupation', IV=['Gender', 'Nationality', 'Occupation'],
                MED=['Education', 'Experience', 'Languages'], DV='Score',
                control='female_non-european_surgeon', out_path=out_path, pos_th=0, run=0)

# generate fair data for the candidates
df_train_count, df_test_count = model.generate_fair_data(df_train, df_test)

# rank the candidates by the debiased score
df_train_reranked = df_train_count.groupby('Occupation').apply(lambda x: x.sort_values(by='Score_fair', ascending=False)).reset_index(drop=True)
df_test_reranked = df_test_count.groupby('Occupation').apply(lambda x: x.sort_values(by='Score_fair', ascending=False)).reset_index(drop=True)

################################ Example LFR ################################
# Output path to save the data and the intermediary steps
out_path = './FairInterventions/out/LFR'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# init pre-processing fairness intervention
model = LearningFairRepresentations(query_col='Occupation', sensitive_col='Gender', score_col='Score', features_col=['Education', 'Experience', 'Languages'],
                                    k = 2, Ax=1, Ay=1, Az=1, out_path=out_path, pos_th=0, run=0, model_occ=True)
# generate fair data for the candidates
df_train_count, df_test_count = model.generate_fair_data(df_train, df_test)

# rank the candidates by the debiased score
df_train_reranked = df_train_count.groupby('Occupation').apply(lambda x: x.sort_values(by='Score_fair', ascending=False)).reset_index(drop=True)
df_test_reranked = df_test_count.groupby('Occupation').apply(lambda x: x.sort_values(by='Score_fair', ascending=False)).reset_index(drop=True)

################################ Example iFair ################################
# Output path to save the data and the intermediary steps
out_path = './FairInterventions/out/iFair'
if not os.path.exists(out_path):
    os.makedirs(out_path)

# init pre-processing fairness intervention
model = iFairRanking(query_col='Occupation', sensitive_col='Gender', score_col='Score', features_col=['Education', 'Experience', 'Languages'],
                                    k = 2, Ax=1, Az=1, out_path=out_path, pos_th=0, run=0, max_iter=100, nb_restarts=3, batch_size=1000, model_occ=True)

# generate fair data for the candidates
df_train_count, df_test_count = model.generate_fair_data(df_train, df_test)

# rank the candidates by the debiased score
df_train_reranked = df_train_count.groupby('Occupation').apply(lambda x: x.sort_values(by='Score_fair', ascending=False)).reset_index(drop=True)
df_test_reranked = df_test_count.groupby('Occupation').apply(lambda x: x.sort_values(by='Score_fair', ascending=False)).reset_index(drop=True)

