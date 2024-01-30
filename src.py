standard_drug_names = ['acetamin', 'biotene', 'compazine', 'ferrous', 'imdur', 'lidocaine', 'milk of magnesia', 'nystatin', 'prochlorperazine', 'tamsulosin',
'advair diskus', 'bisacodyl', 'coreg', 'flagyl', 'influenza vac', 'lipitor', 'mineral', 'omeprazole', 'promethazine', 'thiamine',
'albumin', 'bumetanide', 'cozaar', 'flomax', 'infuvite', 'lisinopril', 'mineral oil', 'ondansetron', 'propofol', 'ticagrelor',
'albuterol', 'bumex', 'decadron', 'flumazenil', 'insulin', 'lispro', 'mono-sod', 'optiray', 'pulmicort respule', 'tiotropium',
'allopurinol', 'buminate', 'definity', 'fluticasone-salmeterol', 'insulin detemir', 'loratadine', 'morphine', 'oxycodone', 'quetiapine', 'toradol',
'alprazolam', 'calcium carbonate', 'deltasone', 'folic acid', 'iohexol', 'lorazepam', 'motrin', 'pantoprazole', 'refresh p.m. op oint', 'tramadol',
'alteplase', 'calcium chloride', 'dexamethasone', 'furosemide', 'iopamidol', 'losartan', 'mupirocin', 'parenteral nutrition', 'reglan', 'trandate',
'alum hydroxide', 'calcium gluconate', 'dexmedetomidine', 'gabapentin', 'ipratropium', 'maalox', 'nafcillin', 'percocet', 'restoril', 'transde rm-scop',
'ambien', 'cardizem', 'dextrose', 'glargine', 'isosorbide', 'magnesium chloride', 'naloxone', 'phenergan', 'ringers solution', 'trazodone',
'aminocaproic acid', 'carvedilol', 'diazepam', 'glucagen', 'kayciel', 'magnesium hydroxide', 'narcan', 'phenylephrine', 'rocuronium', 'ultram',
'amiodarone', 'catapres', 'digoxin', 'glucagon', 'kayexalate', 'magnesium oxide', 'neostigmine', 'phytonadione', 'roxicodone', 'valium',
'amlodipine', 'cefazolin', 'diltiazem', 'glucose', 'keppra', 'magnesium sulf', 'neostigmine methylsulfate', 'piperacillin', 'sennosides', 'vancomycin',
'anticoagulant', 'cefepime', 'diphenhydramine', 'glycopyrrolate', 'ketorolac', 'magox', 'neurontin', 'plasmalyte', 'seroquel', 'vasopressin',
'apresoline', 'ceftriaxone', 'diprivan', 'guaifenesin', 'klonopin', 'medrol', 'nexterone', 'plavix', 'sertraline', 'ventolin',
'ascorbic acid', 'cephulac', 'docusate', 'haldol', 'labetalol', 'meperidine', 'nicardipine', 'pneumococcal', 'simethicone', 'vitamin',
'aspart', 'cetirizine', 'dopamine', 'haloperidol', 'lactated ringer', 'meropenem', 'nicoderm', 'pnu-immune-23', 'simvastatin', 'warfarin',
'aspirin', 'chlorhexidine', 'ecotrin', 'heparin', 'lactulose', 'merrem', 'nicotine', 'polyethylene glycol', 'sodium bicarbonate', 'xanax',
'atenolol', 'ciprofloxacin', 'enoxaparin', 'humulin', 'lanoxin', 'metformin', 'nitro-bid', 'potassium chloride', 'sodium chloride', 'zestril',
'atorvastatin', 'cisatracurium', 'ephedrine', 'hydralazine', 'lantus', 'methylprednisolone', 'nitroglycerin', 'potassium phosphate', 'sodium phosphate', 'zocor',
'atropine', 'citalopram', 'epinephrine', 'hydrochlorothiazide', 'levaquin', 'metoclopramide', 'nitroprusside', 'pravastatin', 'polystyrene sulfonate', 'zolpidem',
'atrovent', 'clindamycin', 'etomidate', 'hydrocodone', 'levemir', 'metoprolol', 'norco', 'precedex', 'spironolactone', 'zosyn',
'azithromycin', 'clonazepam', 'famotidine', 'hydrocortisone', 'levetiracetam', 'metronidazole', 'norepinephrine', 'prednisone', 'sublimaze',
'bacitracin', 'clonidine', 'fat emulsion', 'hydromorphone', 'levofloxacin', 'midazolam', 'normodyne', 'prilocaine', 'succinylcholine',
'bayer chewable', 'clopidogrel', 'fentanyl', 'ibuprofen', 'levothyroxine', 'midodrine', 'norvasc', 'prinivil', 'tacrolimus']

len(standard_drug_names)

import pandas as pd
import re

drugs_df = pd.read_csv('eicu_drug_comp565.csv')
ventilator_df = pd.read_csv('eicu_ventilator_comp565.csv')
sepsis_df = pd.read_csv('eicu_sepsis_comp565.csv')
mortality_df = pd.read_csv('eicu_mortality_comp565.csv')

drugs_df = drugs_df.fillna('Unknown')

drug_name_dict = {}

for drug in standard_drug_names:
    aliases = drugs_df[drugs_df['drugname'].str.contains(drug)]['drugname'].unique()

    for alias in aliases:
        drug_name_dict[alias] = drug

drugs_df['drugname'] = drugs_df['drugname'].replace(drug_name_dict)

def disregard_dosage(s):
  matched = re.search(r"\d",s)
  if matched:
    index = matched.start()
    s=s[:index]
  return s

drugs_df['drugname'] = drugs_df['drugname'].apply(disregard_dosage)

drugs_df

"""## Time-Dependent Topic Model"""

import numpy as np
from scipy.stats import dirichlet

# Number of topics
K = 9

# Number of patients
D = len(drugs_df['patientunitstayid'].unique())

# Initialize α
alpha = np.ones(K)

theta = {}

patient_ids = drugs_df['patientunitstayid'].unique()

for d in patient_ids:
    print(d)
    Nd = len(drugs_df[drugs_df['patientunitstayid'] == d])
    theta[d] = np.zeros((Nd, K))

V = len(drugs_df['drugname'].unique())

phi = np.random.dirichlet([1]*V, size=K)

patient_id_mapping = {old_id: new_id for new_id, old_id in enumerate(patient_ids)}

reverse_patient_id_mapping = {new_id: old_id for old_id, new_id in patient_id_mapping.items()}

drugs_df['patientunitstayid'] = drugs_df['patientunitstayid'].map(patient_id_mapping)

V = len(np.unique(drugs_df['drugname']))
ndk = np.zeros((D, K))
nkv = np.zeros((K, V))
nk = np.zeros(K)

unique_drugs = drugs_df['drugname'].unique()

drug_index_mapping = {drug: index for index, drug in enumerate(unique_drugs)}

drugs_df['drugname'] = drugs_df['drugname'].map(drug_index_mapping)

drugs_df

N_ITER = 1
alpha = np.ones(K)
beta = 1.0

z = {}
x = {}

for iter in range(N_ITER):
    print(f'iter{iter}')
    for d in range(D):
        print(d)
        patient_data = drugs_df[drugs_df['patientunitstayid'] == d]
        patient_data = patient_data.sort_values('drugstartoffset')
        Nd = len(patient_data)
        z[d] = np.zeros(Nd, dtype=int)
        x[d] = np.zeros(Nd, dtype=int)
        theta[d] = np.zeros((Nd, K))

        for i in range(Nd):
            k_old = z[d][i]
            v = x[d][i]
            v = patient_data.iloc[i]['drugname']

            if ndk[d, k_old] > 0:
                ndk[d, k_old] -= 1
            if nkv[k_old, v] > 0:
                nkv[k_old, v] -= 1
            if nk[k_old] > 0:
                nk[k_old] -= 1

            # (a) Sample Topic Mixture
            if i == 0:
                theta[d][i, :] = dirichlet.rvs(alpha)
            else:
                theta[d][i, :] = dirichlet.rvs(theta[d][i-1, :]+1e-10)

            # Compute the conditional distribution
            p = (ndk[d, :] + alpha) * (nkv[:, v] + beta) / (nk + V * beta)

            # (b) Sample Topic
            k_new = np.random.choice(range(K), p=p/p.sum())
            z[d][i] = k_new

            # (c) Sample a Drug
            x[d][i] = np.random.choice(range(V), p=phi[z[d][i], :])

            ndk[d, k_new] += 1
            nkv[k_new, v] += 1
            nk[k_new] += 1

    theta = {d: (ndk[d, :] + alpha) / (ndk[d, :].sum() + K * alpha) for d in range(D)}
    phi = (nkv + beta) / (nkv.sum(axis=1, keepdims=True) + V * beta)

    # Online Update
    for d in range(D):
        print(f'update{d}')
        for k in range(K):
            for t in range(len(z[d])):
                ndk[d, k] = np.sum(z[d][:t+1] == k)

z[0]

phi.shape

x[0]

theta[0]

theta[33915]

V

from gensim.corpora import Dictionary
from gensim.models import LdaModel

documents = []
drugs_df_str = drugs_df.copy()
drugs_df_str['drugname'] = drugs_df_str['drugname'].astype(str)

documents = []
for patient_id in drugs_df_str['patientunitstayid'].unique():
    print(patient_id)
    patient_drugs = drugs_df_str[drugs_df_str['patientunitstayid'] == patient_id].sort_values('drugstartoffset')['drugname'].tolist()
    documents.append(patient_drugs)

len(documents[0])

dictionary = Dictionary(documents)

corpus = [dictionary.doc2bow(doc) for doc in documents]

model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=9,
    iterations=1,
    passes=20,
)

topics = model.show_topics(num_topics=10, num_words=10)

phi_standard = model.get_topics()
theta_standard = []

for doc in corpus:
    doc_topics = model.get_document_topics(doc, minimum_probability=0)
    doc_topics = [prob for _, prob in doc_topics]
    theta_standard.append(doc_topics)

theta_standard = np.array(theta_standard)

phi_standard.shape

theta_standard.shape

theta_standard[33915]

drug_index_mapping

reverse_drug_id_mapping = {new_id: old_name for old_name, new_id in drug_index_mapping.items()}

phi_df = pd.DataFrame(phi).transpose()

for k in range(K):
    topic_data = phi_df[k]
    sorted_drugs = topic_data.sort_values(ascending=False)

    top_drugs = sorted_drugs[:3]

    print(f"Top 3 drugs for Topic {k}:")
    for drug, value in top_drugs.items():
        print(f"Drug: {reverse_drug_id_mapping[drug]}, Value: {value}")
    print("\n")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


reverse_drug_id_mapping = {new_id: old_name for old_name, new_id in drug_index_mapping.items()}

phi_df_copy = phi_df.copy()

phi_df_copy.index = phi_df_copy.index.map(reverse_drug_id_mapping)

plt.figure(figsize=(5, 10))
sns.heatmap(phi_df_copy, cmap='YlGnBu', cbar_kws={'shrink': 0.5})

plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

phi_standard_df = pd.DataFrame(phi_standard).transpose()

phi_standard_df_copy = phi_standard_df.copy()

phi_standard_df_copy.index = phi_standard_df_copy.index.map(reverse_drug_id_mapping)

plt.figure(figsize=(5, 10))
sns.heatmap(phi_standard_df_copy, cmap='YlGnBu', cbar_kws={'shrink': 0.5})

plt.show()

drug_counts = drugs_df.groupby('patientunitstayid')['drugname'].nunique()

drug_counts_df = drug_counts.reset_index()

drug_counts_df.columns = ['patientunitstayid', 'num_drugs']

drug_counts_df

import matplotlib.pyplot as plt
import numpy as np

theta_array = np.zeros((D, K))
for d in range(D):
    theta_array[d] = theta[d]

top_patients = np.argmax(theta_array, axis=0)

for k in range(K):
    top_patient = top_patients[k]
    topic_assignments = z[top_patient]
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(topic_assignments)), topic_assignments)
    plt.yticks(range(11))
    plt.xlabel('Time points')
    plt.ylabel('Topic assignments')
    plt.title(f'Topic {k} assignments for top patient {reverse_patient_id_mapping[top_patient]}')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

theta_array = np.zeros((D, K))
for d in range(D):
    theta_array[d] = theta[d]

top_patients = np.argmax(theta_array, axis=0)
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

for k in range(K):
    top_patient = top_patients[k]
    topic_assignments = z[top_patient]
    ax = axs[k // 3, k % 3]
    ax.scatter(range(len(topic_assignments)), topic_assignments)
    ax.set_yticks(range(11))
    ax.set_xlabel('Time points')
    ax.set_ylabel('Topic assignments')
    ax.set_title(f'Topic {k} assignments for top patient {reverse_patient_id_mapping[top_patient]}')

plt.tight_layout()
plt.show()

max_values = np.max(theta_array, axis=0)

for k in range(K):
    top_patient = top_patients[k]
    print(f"Top patient for Topic {k}: {reverse_patient_id_mapping[top_patient]}, Max value: {max_values[k]}")

drugs_df = pd.read_csv('eicu_drug_comp565.csv')
ventilator_df = pd.read_csv('eicu_ventilator_comp565.csv')
sepsis_df = pd.read_csv('eicu_sepsis_comp565.csv')
mortality_df = pd.read_csv('eicu_mortality_comp565.csv')

ventilator_df['ventilator'] = 1
sepsis_df['sepsis'] = 1
mortality_df['death'] = 1

patient_ids = drugs_df['patientunitstayid'].unique()

labels = np.zeros((len(patient_ids), 3))

for i, patient_id in enumerate(patient_ids):
    ventilator = int(patient_id in ventilator_df['patientunitstayid'].values)
    sepsis = int(patient_id in sepsis_df['patientunitstayid'].values)
    death = int(patient_id in mortality_df['patientunitstayid'].values)
    labels[i] = [ventilator, sepsis, death]

labels

patient_ids

correlation_matrix = np.zeros((K, 3))

for d in range(D):
    topic = z[d][-1]
    features = labels[d]
    correlation_matrix[topic] += features

correlation_matrix = correlation_matrix / correlation_matrix.sum(axis=0, keepdims=True)

correlation_matrix

import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, xticklabels=['sepsis', 'ventilator', 'death'], yticklabels=range(K), cmap='viridis_r')
plt.xlabel('Features')
plt.ylabel('Topics')
plt.show()