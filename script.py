import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats


def preprocess(og_data, dv):
	y_all = og_data[dv] #Loading the dependent variable (dv) aka the variable to be predicted
	data_all = og_data.drop(dv,1) #remove the dv from the main dataset 
	data_all = data_all[data_all.columns[data_all.isnull().sum() < 1000]] #drop variables with over a 1000 NaNs 
	return data_all,y_all

def multicollin(preprocessed_data):
	corr_matrix = preprocessed_data.corr().abs() 
	to_drop = []
	upper_train = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
	.unstack().sort_values(ascending=False)) #get upper matrix of corr matrix and stack the values in increasing order
	upper_train = upper_train[upper_train > 0.7] #set threshold
	for val in upper_train.index.values:
		for j in val:
			to_drop.append(j)
	data_all= preprocessed_data.drop(columns=to_drop) #remove variables that were highly correlated
	return data_all

def getdummies(preprocessed_data):
	cat_columns = preprocessed_data.select_dtypes(include=['object']).columns #pick categorical variables
	data = pd.get_dummies(preprocessed_data,prefix='dummy',columns=cat_columns)# convert these into dummy variables
	return data

def replacenan(dummies_data):
	impute =  SimpleImputer(missing_values=np.nan, strategy='mean') #replacing missing data with substituted mean values
	imputed_data = impute.fit(dummies_data)
	SimpleImputer()
	imputed_data = impute.transform(dummies_data)
	data = pd.DataFrame(imputed_data, columns = dummies_data.columns)
	return data

def standardize(replaced_data):
	sc = StandardScaler() # mean normalization using z = (x - u) / s, where z is normalized value, x is the original, u is the mean and s is the stdev
	X = sc.fit_transform(replaced_data)
	return X

def split(standardized_data,y_preprocessed,testsize):
	y = y_preprocessed
	y = y.values.reshape((len(y),1)) #reshape to be a vector of shape (features,1)
	X_train, X_test, y_train, y_test = train_test_split(standardized_data, y, test_size=testsize)
	return X_train, X_test, y_train, y_test


def PCA(train_data):
	cov_mat = np.cov(train_data.T) #make covariance matrix of features in dataset
	U, S, V = np.linalg.svd(cov_mat) #singular value decomposition of covariance matrix - # U & V hold eigenvectors and the diagonal of S holds eigenvalues
	rho = (S*S) / (S*S).sum() #using S matrix from svd to find optimal K
	threshold = 0.99
	cumsum = np.cumsum(rho)
	K = np.argwhere(cumsum > threshold) #no of principals components that explain 99% of the variability 
	U_reduced = U[:,:len(K)] # only choosing K no of columns from the U matrix
	Z = np.dot(train_data,U_reduced) #transforming the original dataset into the reduced dimensions
	return U_reduced, Z


def re_shape(final_data):
	X =np.c_[np.ones((len(final_data),1)),final_data] #Add bias term
	theta = np.zeros((X.shape[1],1)) #initalize theta to zeros of size (features,1) or could also do np.random.randn(X.shape[1],1)
	return X,theta


def Cost(X_train,y_train,theta):
	m = len(y_train)
	pred = np.dot(X_train,theta)
	loss = pred-y_train
	J = (1/(2*m))*np.sum(np.square(loss))
	return J

def gradientDes(X_train,y_train,theta,alpha,num_iters):
	m = len(y_train)
	J_history = np.zeros(num_iters)
	for i in range(num_iters):
		pred = X_train.dot(theta)
		loss = pred-y_train
		gradient = X_train.T.dot(loss)
		theta = theta - (1/m)*alpha*(gradient)
		J_history[i] = Cost(X_train, y_train, theta)
		
	return theta, J_history


def pred(X_test, params):
	return X_test.dot(params)

if __name__ == '__main__':
	path = '/Users/ifrahkhanyaree/Desktop/Kurzarbeit/Project/housing_PCA_LinReg'
	train_df = pd.DataFrame(pd.read_csv(path+'/train.csv'))
	preprocessing, y_all = preprocess(train_df, "SalePrice")


	learningrates = [0.003,0.03,0.05]
	lst_numofiters = [500,1000,10000]
	testsizes = [0.4,0.3,0.2]
	r2_scores_PCA = []
	r2_scores_NoPCA = []
	
	#PCA
	add_dummies_PCA = getdummies(preprocessing)
	imputing_PCA = replacenan(add_dummies_PCA)
	featurenorm_PCA = standardize(imputing_PCA)



	#NoPCA
	multicoll = multicollin(preprocessing)
	add_dummies = getdummies(multicoll)
	imputing = replacenan(add_dummies)
	featurenorm = standardize(imputing)




	for sizes in testsizes:
		for learningrate in learningrates:
			X_train, X_test, y_train, y_test = split(featurenorm,y_all,sizes)
			data,theta = re_shape(X_train)
			theta_trained, J_history = gradientDes(data, y_train, theta, learningrate, 1000)
			X_test_reshaped = np.c_[np.ones((len(X_test),1)),X_test]
			predictions = pred(X_test_reshaped,theta_trained)
			r2 = r2_score(y_test, predictions)
			r2_scores_NoPCA.append(r2)

			X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = split(featurenorm_PCA,y_all,sizes)
			U_reduced, Z = PCA(X_train_PCA)
			data_PCA,theta_PCA = re_shape(Z)
			theta_trained_PCA, J_history_PCA = gradientDes(data_PCA, y_train_PCA, theta_PCA, learningrate, 1000)
			X_test_PCA = np.dot(X_test_PCA,U_reduced)
			X_test_PCA_reshaped = np.c_[np.ones((len(X_test_PCA),1)),X_test_PCA]
			predictions_PCA = pred(X_test_PCA_reshaped,theta_trained_PCA)
			r2_PCA = r2_score(y_test_PCA, predictions_PCA)
			r2_scores_PCA.append(r2_PCA)
	
	X = ['0.003_0.4','0.003_0.3','0.003_0.2','0.03_0.4','0.03_0.3','0.03_0.2','0.05_0.4','0.05_0.3','0.05_0.2']
	pos = np.arange(len(X))
	width = 0.3
	plt.bar(pos,r2_scores_NoPCA,width,color='magenta')
	plt.bar(pos+width,r2_scores_PCA,width,color='yellow')
	plt.xticks(pos+width/2,X,rotation=20)
	plt.title('Multiple Linear Regression with varying learning rates & test sizes')
	plt.xlabel('Learning rates & Test sizes')
	plt.ylabel('R2 score')
	plt.legend(['NoPCA','PCA'],loc='best')
	plt.show()

t2, pv = stats.ttest_ind(r2_scores_NoPCA,r2_scores_PCA)
print(t2,pv)	




