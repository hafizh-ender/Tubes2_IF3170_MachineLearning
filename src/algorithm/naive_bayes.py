import numpy as np
import pandas as pd

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes Classifier
    
    This classifier predicts the class of an input sample based on the probability of the sample
    belonging to each class, computed using the Gaussian probability density function.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    class_priors_ : dict
        The prior probabilities of each class.
    class_means_ : dict
        The mean values of each feature for each class.
    class_variances_ : dict
        The variance values of each feature for each class.
    feature_names_ : dict
        The feature names of the training data.
    """
    def __init__(self):
        self.class_priors_ = {}
        self.class_means_ = {}
        self.class_variances_ = {}
        self.feature_names_ = []
        
    def fit(self, X, y):
        """
        Fit the classifier with the training data. Compute the class priors, means, and variances.
        
        Parameters
        ----------
        X : list of shape (n_samples, n_features)
            The training data.
        y : list of shape (n_samples,)
            The target labels corresponding to the training data. Can contain strings or other types.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        
         # Save the feature names if X is a DataFrame, else fill with numbers
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [i for i in range(len(X[0]))]
        
        # If X are DataFrame, convert them to a list
        if isinstance(X, pd.DataFrame):
            X = X.values.tolist()
            
        # Combine the features and labels into a single DataFrame
        data = pd.DataFrame(X)
        data['target_variable_'] = y
        
        # Compute the class priors
        self.class_priors_ = data['target_variable_'].value_counts(normalize=True).to_dict()
        
        # Compute the class means and variances
        for label in self.class_priors_:
            subset = data[data['target_variable_'] == label].drop(columns='target_variable_')
            self.class_means_[label] = subset.mean().to_dict()
            self.class_variances_[label] = subset.var().to_dict()
            
    def predict(self, X):
        """
        Predict the class labels for the input data.
        
        Parameters
        ----------
        X : list of shape (n_samples, n_features)
            The input data to classify.
        
        Returns
        -------
        y_pred : list of shape (n_samples,)
            The predicted class labels for each input sample.
        """        
        # Check if the classifier has been fitted
        if not self.class_priors_:
            raise ValueError("The classifier has not been fitted yet.")
        
        # Check if X is a DataFrame and convert it to a list if it is
        if isinstance(X, pd.DataFrame):
            X = X.values.tolist()
            
        # Check if X has the same number of features as the training data
        if len(X[0]) != len(self.feature_names_):
            raise ValueError("The input data has a different number of features than the training data.")
        
        
        y_pred = []
        for x in X:
            # Compute the probabilities of each class
            class_probs = {}
            for label in self.class_priors_:
                class_probs[label] = self._compute_class_probability(x, label)
                
            # Determine the class with the highest probability
            y_pred.append(max(class_probs, key=class_probs.get))
            
        return y_pred
    
    def _compute_class_probability(self, x, label):
        """
        Compute the probability of a sample belonging to a specific class.
        
        Parameters
        ----------
        x : list or tuple of shape (n_features,)
            The input sample.
        label : any
            The class label for which to compute the probability.
        
        Returns
        -------
        class_prob : float
            The computed probability of the sample belonging to the specified class.
        """
        class_prob = self.class_priors_[label]
        
        for i, value in enumerate(x):
            mean = self.class_means_[label][i]
            variance = self.class_variances_[label][i]
            class_prob *= self._gaussian_pdf(value, mean, variance)
            
        return class_prob
    
    def _gaussian_pdf(self, x, mean, variance):
        """
        Compute the Gaussian probability density function.
        
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
        pdf : float
            The computed probability density function value for the input value.
        """
        return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))
    
    def save(self, filename='naive_bayes.txt'):
        """
        Save the classifier model to a text file
        
        
        Parameters
        ----------
        filename : str
            The name of the file to save the model to.
        """
        with open(filename, 'w') as file:
            file.write('class_priors\n')
            for key, value in self.class_priors_.items():
                file.write(f'{key}: {value}\n')
                
            file.write('class_means\n')
            for key, value in self.class_means_.items():
                file.write(f'{key}: {value}\n')
                
            file.write('class_variances\n')
            for key, value in self.class_variances_.items():
                file.write(f'{key}: {value}\n')
                
            file.write('feature_names\n')
            file.write(f'{self.feature_names_}\n')
            
            
    def load(self, filename='naive_bayes.txt'):
        """
        Load the classifier model from a text file.
        
        Parameters
        ----------
        filename : str
            The name of the file containing the saved model.
        """
        with open(filename, 'r') as file:
            data = file.read()

        lines = data.strip().split('\n')
        result = {}
        current_key = None

        for line in lines:
            if line in ['class_priors', 'class_means', 'class_variances', 'feature_names']:
                current_key = line
                result[current_key] = {}
            elif current_key == 'feature_names':
                result[current_key] = eval(line)
            else:
                key, value = line.split(': ', 1)
                if current_key in ['class_means', 'class_variances']:
                    result[current_key][int(key)] = eval(value)
                else:
                    result[current_key][int(key)] = float(value)

        self.class_priors_ = result.get('class_priors', {})
        self.class_means_ = result.get('class_means', {})
        self.class_variances_ = result.get('class_variances', {})
        self.feature_names_ = result.get('feature_names', [])