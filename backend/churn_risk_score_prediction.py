import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os 

print(os.getcwd())



"""### Getting Data"""

df = pd.read_csv(r'train.csv')
# df

"""### Data Preprocessing"""

df.info()

df = df.drop(['Name', 'referral_id'], axis=1)  # dropping unnecessary columns
# df

label_encoder = preprocessing.LabelEncoder()  # encoding data

a = df.columns
for i in a[:-1]:
  df[i] = df[i].astype('|S')
  df[i] = label_encoder.fit_transform(df[i])
# df

df['churn_risk_score'].isnull().any()  # checking for and removing records with null values

df = df.dropna(axis = 0, how ='any')

df.isnull().any()

# df

"""### Data Visualization"""

# checking the distribution of outcomes
sns.countplot(x = 'churn_risk_score', data = df)

"""### Checking Variance"""

# df.columns

# checking variance
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = df[['customer_id', 'age', 'gender', 'security_no', 'region_category',
       'membership_category', 'joining_date', 'joined_through_referral',
       'preferred_offer_types', 'medium_of_operation', 'internet_option',
       'last_visit_time', 'days_since_last_login', 'avg_time_spent',
       'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet',
       'used_special_discount', 'offer_application_preference',
       'past_complaint', 'complaint_status', 'feedback', 'churn_risk_score']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns

# vif

"""### VIF is less than 10 for all the attributes, hence, we can keep them all.

### Splitting Data for Training and Testing
"""

data = df.values
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 
# splitting in the ratio 80:20

"""### Decision Tree Model"""

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

"""### Making Predictions"""

pred = clf.predict(X_test)

"""### Checking Accuracy"""

score = clf.score(X_test, y_test)
# score

"""### Predictions are 71% accurate.

### Visualizing the Decision Tree
"""

s = plt.figure(figsize=(20,10))
tree.plot_tree(clf, max_depth=2, filled=True, fontsize=10)
plt.title("Decision Tree for Customer Churn Data", fontsize=30)
plt.show()

"""### Getting the pkl File"""

#from sklearn.externals import joblib
#import sklearn.external.joblib as extjoblib
import joblib

# saving the model as a pickle in a file
joblib.dump(clf, 'ChurnRiskScorePrediction.pkl')


"""### GUI"""

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = tk.Tk()

canvas = FigureCanvasTkAgg(s, master = root)

root.title('Plotting in Tkinter')
root.geometry("1300x700")
canvas.draw()
canvas.get_tk_widget().pack()

root.mainloop()

