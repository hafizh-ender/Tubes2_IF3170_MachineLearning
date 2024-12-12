import numpy as np
import pandas as pd

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes Classifier
    
    This classifier predicts the class of an input sample based on the probability of the sample
    belonging to each class, computed using the Gaussian probability density function.
    
    Parameters
    ----------
    var_smoothing : float, default=1e-9
    Portion of the largest variance of all features that is added to variances for calculation stability.
    
    Attributes
    ----------
    class_priors_ : numpy.ndarray
        The prior probabilities of each class.
    mean_ : numpy.ndarray
        The mean values of each feature for each class.
    var_ : numpy.ndarray
        The variance values of each feature for each class.
    feature_names_in_ : list
        The feature names of the training data.
    """
    def __init__(self, var_smoothing=1e-9):
        self.classes_ = []
        self.feature_names_in_ = []
        self.class_prior_ = []
        self.mean_ = []
        self.var_ = []
        self.var_smoothing_ = var_smoothing
        
    def fit(self, X, y):
        """
        Fit the classifier with the training data. Compute the class priors, means, and variances.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : array-like of shape (n_samples,)
            The target labels corresponding to the training data. Can contain strings or other types.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        
         # If X is not a DataFrame, convert it to one
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Store the feature names
        self.feature_names_in_ = np.array(X.columns)
        
        # Combine the input data and target labels
        data = X.copy()
        data['target_variable_'] = y
        
        # Compute the class priors and store the classes
        unique_classes, counts = np.unique(data['target_variable_'], return_counts=True)
        
        self.classes_ = unique_classes
        self.class_prior_ = counts / counts.sum()
        
        # Compute the class means and variances
        grouped_data = data.groupby('target_variable_')
        self.mean_ = grouped_data.mean().values
        self.var_ = grouped_data.var().values
        
        # Add the variance smoothing value to the variance values
        self.var_ += self.var_smoothing_ * np.var(X, axis=0).max()

            
    def predict(self, X):
        """
        Predict the class labels for the input data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to classify.
        
        Returns
        -------
        y_pred : numpy.ndarray of shape (n_samples,)
            The predicted class labels for each input sample.
        """    
        # Check if the classifier has been trained by checking if the class priors is empty
        if not len(self.class_prior_):
            raise ValueError("The classifier has not been trained.")
        
        # Check if X is a DataFrame and convert it to a list if it is
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Check if X has the same number of features as the training data
        if X.shape[1] != len(self.feature_names_in_):
            raise ValueError("The input data has a different number of features than the training data.")
        
        # Use log probabilities to avoid underflow
        log_class_prior = np.log(self.class_prior_)
        log_class_prob = np.zeros((X.shape[0], len(self.classes_)))
        
        for i in range(len(self.classes_)):
            log_class_prob[:, i] = np.sum(self._log_gaussian_pdf(X, self.mean_[i, :], self.var_[i, :]), axis=1) + log_class_prior[i]
        
        y_pred = self.classes_[np.argmax(log_class_prob, axis=1)]
        
        return y_pred
    
    def _log_gaussian_pdf(self, x, mean, var):
        """
        Compute the Gaussian probability density function in log space.
        
        Parameters
        ----------
        x : float
            The input value.
        mean : float
            The mean value of the distribution.
        variance : float
            The variance value of the distribution.
        
        Returns
        -------
        log_pdf : float
            The computed probability density function value for the input value in log space.
        """
        # Compute the Gaussian probability density function in log space
        log_coeff = -0.5 * np.log(2.0 * np.pi * var)
        log_exponent = -0.5 * ((x - mean) ** 2) / (var)
        
        return log_coeff + log_exponent
    
    def save(self, filename='naive_bayes.txt'):
        """
        Save the classifier model to a text file
        
        Parameters
        ----------
        filename : str
            The name of the file to save the model to.
        """
        with open(filename, 'w') as file:
            file.write('class_prior:')
            file.write(f'{self.class_prior_.tolist()}\n')
            
            file.write('mean:')
            file.write(f'{self.mean_.tolist()}\n')
            
            file.write('var:')
            file.write(f'{self.var_.tolist()}\n')
            
            file.write('feature_names_in:')
            file.write(f'{self.feature_names_in_.tolist()}\n')
            
            
    def load(self, filename='naive_bayes.txt'):
        """
        Load the classifier model from a text file.
        
        Parameters
        ----------
        filename : str
            The name of the file containing the saved model.
        """
        with open(filename, 'r') as file:
            lines = file.readlines()
            
        # Parsing the data
        parsed_data = {}
        for line in lines:
            key, value = line.split(":", 1)
            parsed_data[key] = eval(value)

        # Accessing parsed values
        self.class_prior_ = np.array(parsed_data['class_prior'])
        self.mean_ = np.array(parsed_data['mean'])
        self.var_ = np.array(parsed_data['var'])
        self.feature_names_in_ = np.array(parsed_data['feature_names_in'])